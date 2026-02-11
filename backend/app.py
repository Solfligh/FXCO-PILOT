import base64
import json
import os
import re
import time
import hmac
import hashlib
from collections import deque

import requests
from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)

# ============================================================
# ENV NOTES
# ============================================================
# OpenAI:
#   OPENAI_API_KEY=...
#
# Twelve Data:
#   TWELVE_DATA_API_KEY=...
#
# Paystack:
#   PAYSTACK_SECRET_KEY=sk_live_or_test_xxx
#   PAYSTACK_PUBLIC_KEY=pk_live_or_test_xxx   # optional
#   PAYSTACK_CALLBACK_URL=https://fxco-pilot.solfightech.org/
#
# Payment gate toggle (supports both):
#   REQUIRE_PAYMENT=true/false
#   PAYSTACK_REQUIRE_PAYMENT=1/0
#
# Amount (supports both):
#   PAYSTACK_AMOUNT=1000000      # MINOR units (kobo) preferred (matches your Render screenshot)
#   OR PAYSTACK_AMOUNT_NGN=10000 # MAJOR units (naira) fallback -> converted to minor
#
# Currency (supports both):
#   PAYSTACK_CURRENCY=NGN
#   PAYSTACK_CURRENCY_CODE=NGN (optional alt)
#
# Trial:
#   TRIAL_DAYS=14
#
# Resend (email OTP):
#   RESEND_API_KEY=...
#   RESEND_FROM=FXCO-PILOT <no-reply@solfightech.org>
#   AUTH_SIGNING_SECRET=long_random_string
#   RESEND_OTP_TTL_SECONDS=600
#
# Rate limits:
#   FREE_LIMIT_60S=5
#   FREE_LIMIT_24H=50
#   PAID_LIMIT_60S=60
#   PAID_LIMIT_24H=2000
#
# Optional guest throttles (defaults used if missing):
#   GUEST_LIMIT_60S=2
#   GUEST_LIMIT_24H=10
# ============================================================

# ==========================
# Simple in-memory cache
# ==========================
_CACHE = {}
_CACHE_TTL = {}

def _cache_get(key):
    if key in _CACHE and time.time() < _CACHE_TTL.get(key, 0):
        return _CACHE[key]
    return None

def _cache_set(key, value, ttl=60):
    _CACHE[key] = value
    _CACHE_TTL[key] = time.time() + ttl


# ==========================
# Helpers: env access
# ==========================
def _env_first(*keys, default=""):
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default

def _env_bool(*keys, default=False):
    v = _env_first(*keys, default=("1" if default else "0")).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _env_int(*keys, default=0, minv=None, maxv=None):
    raw = _env_first(*keys, default=str(default))
    try:
        n = int(str(raw).strip())
    except Exception:
        n = int(default)
    if minv is not None:
        n = max(minv, n)
    if maxv is not None:
        n = min(maxv, n)
    return n


# ==========================
# Rate limiting (tiered)
#   - Guest: very low (to slow spam)
#   - Trial: uses FREE_LIMIT_*
#   - Paid: uses PAID_LIMIT_*
# Applied ONLY to /api/analyze
# ==========================
_SHORT_WINDOW_SECONDS = 60
_DAY_WINDOW_SECONDS = 24 * 60 * 60

# key: identity -> deque[timestamps]
_RATE_SHORT = {}
_RATE_DAY = {}


def _client_ip():
    """
    Try to get the real client IP behind proxies.
    X-Forwarded-For is usually: "client, proxy1, proxy2"
    """
    xff = request.headers.get("X-Forwarded-For", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


# ==========================
# OpenAI client (lazy init)
# ==========================
def _get_openai_client():
    api_key = _env_first("OPENAI_API_KEY", default="").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# ==========================
# Twelve Data
# ==========================
TWELVE_DATA_API_KEY = _env_first("TWELVE_DATA_API_KEY", default="").strip()
TD_BASE = "https://api.twelvedata.com"


# ==========================
# Paystack config + state (in-memory)
# ==========================
PAYSTACK_SECRET_KEY = _env_first("PAYSTACK_SECRET_KEY", default="").strip()
PAYSTACK_PUBLIC_KEY = _env_first("PAYSTACK_PUBLIC_KEY", default="").strip()
PAYSTACK_CURRENCY = _env_first("PAYSTACK_CURRENCY", "PAYSTACK_CURRENCY_CODE", default="NGN").strip().upper()
PAYSTACK_BASE = "https://api.paystack.co"

def _require_payment():
    # supports both env names
    return _env_bool("REQUIRE_PAYMENT", "PAYSTACK_REQUIRE_PAYMENT", default=True)

def _access_days():
    return _env_int("PAYSTACK_ACCESS_DAYS", default=30, minv=1, maxv=3650)

def _trial_days():
    return _env_int("TRIAL_DAYS", default=14, minv=1, maxv=365)

def _paystack_amount_minor():
    """
    Preferred: PAYSTACK_AMOUNT is already MINOR units (kobo) (matches your Render screenshot).
    Fallback: PAYSTACK_AMOUNT_NGN is MAJOR units -> convert to minor.
    """
    if _env_first("PAYSTACK_AMOUNT", default="").strip() != "":
        amt = _env_int("PAYSTACK_AMOUNT", default=0, minv=0)
        return amt

    major = _env_int("PAYSTACK_AMOUNT_NGN", default=10000, minv=0)
    return major * 100

# We store:
#  - pending reference -> client_id
#  - access client_id -> paid_until_epoch
_PENDING_REF = {}  # ref -> {"client_id": str, "created": epoch}
_ACCESS = {}       # client_id -> {"paid_until": epoch, "last_ref": str, "updated": epoch}

def _get_client_id_header():
    cid = (request.headers.get("X-FXCO-Client") or "").strip()
    if cid:
        return cid[:128]
    return ""

def _now():
    return time.time()

def _cleanup_payment_state():
    now = _now()
    for ref in list(_PENDING_REF.keys()):
        if now - _PENDING_REF[ref].get("created", now) > 24 * 60 * 60:
            _PENDING_REF.pop(ref, None)
    for cid in list(_ACCESS.keys()):
        paid_until = _ACCESS[cid].get("paid_until", 0)
        if paid_until and now > (paid_until + 7 * 24 * 60 * 60):
            _ACCESS.pop(cid, None)

def _paystack_headers():
    return {
        "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def _paystack_ok_enabled():
    """
    If payment is not required => OK.
    If payment is required => need secret key.
    """
    if not _require_payment():
        return True
    return bool(PAYSTACK_SECRET_KEY)

def _callback_url():
    cb = _env_first("PAYSTACK_CALLBACK_URL", default="").strip()
    if cb:
        return cb
    origin = (request.headers.get("Origin") or "").strip()
    if origin:
        return origin
    return request.host_url.rstrip("/")

def _ensure_paystack_success_param(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    if "paystack=success" in u:
        return u
    sep = "&" if "?" in u else "?"
    return f"{u}{sep}paystack=success"

def _email_ok(email: str) -> bool:
    if not email:
        return False
    email = email.strip()
    if len(email) > 254:
        return False
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email))

def _is_client_unlocked(client_id: str) -> bool:
    _cleanup_payment_state()
    if not _require_payment():
        # if payment gate is off, treat as "unlocked"
        return True
    if not client_id:
        return False
    row = _ACCESS.get(client_id)
    if not row:
        return False
    return _now() < float(row.get("paid_until", 0))


# ==========================
# Resend OTP + Trial (in-memory)
# ==========================
RESEND_API_KEY = _env_first("RESEND_API_KEY", default="").strip()
RESEND_FROM = _env_first("RESEND_FROM", default="").strip()
AUTH_SIGNING_SECRET = _env_first("AUTH_SIGNING_SECRET", default="").strip()
OTP_TTL_SECONDS = _env_int("RESEND_OTP_TTL_SECONDS", default=600, minv=60, maxv=3600)

_OTP = {}     # email -> {"code": "123456", "exp": epoch, "attempts": int}
_TRIAL = {}   # email -> {"start": epoch, "end": epoch}

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

def _b64url_decode(s: str) -> bytes:
    s = (s or "").strip()
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

def _sign_token(payload: dict) -> str:
    if not AUTH_SIGNING_SECRET:
        return ""
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    p = _b64url(raw)
    sig = hmac.new(AUTH_SIGNING_SECRET.encode("utf-8"), p.encode("utf-8"), hashlib.sha256).digest()
    return f"{p}.{_b64url(sig)}"

def _verify_token(token: str) -> dict | None:
    if not AUTH_SIGNING_SECRET:
        return None
    t = (token or "").strip()
    if not t or "." not in t:
        return None
    p, s = t.split(".", 1)
    try:
        expected = hmac.new(AUTH_SIGNING_SECRET.encode("utf-8"), p.encode("utf-8"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url(expected), s):
            return None
        payload = json.loads(_b64url_decode(p).decode("utf-8"))
        exp = float(payload.get("exp") or 0)
        if exp and _now() > exp:
            return None
        return payload
    except Exception:
        return None

def _get_auth_email():
    """
    Accepts token in:
      - X-FXCO-Auth: <token>
      - Authorization: Bearer <token>
    Returns lowercased email or "".
    """
    token = (request.headers.get("X-FXCO-Auth") or "").strip()
    if not token:
        authz = (request.headers.get("Authorization") or "").strip()
        if authz.lower().startswith("bearer "):
            token = authz.split(" ", 1)[1].strip()

    payload = _verify_token(token)
    if not payload:
        return ""
    email = (payload.get("email") or "").strip().lower()
    if not _email_ok(email):
        return ""
    return email

def _trial_status(email: str):
    """
    Returns tuple: (active: bool, end_epoch: int | None, started: bool)
    """
    e = (email or "").strip().lower()
    if not e:
        return (False, None, False)

    row = _TRIAL.get(e)
    if not row:
        return (False, None, False)

    end = float(row.get("end") or 0)
    active = _now() < end
    return (active, int(end) if end else None, True)

def _ensure_trial(email: str):
    e = (email or "").strip().lower()
    if not e:
        return None
    if e in _TRIAL:
        return _TRIAL[e]
    start = _now()
    end = start + (_trial_days() * 24 * 60 * 60)
    _TRIAL[e] = {"start": start, "end": end}
    return _TRIAL[e]

def _resend_send_otp(email: str, code: str):
    if not RESEND_API_KEY or not RESEND_FROM:
        return False, "Resend not configured (missing RESEND_API_KEY or RESEND_FROM)."

    subject = "Your FXCO-PILOT verification code"
    html = f"""
    <div style="font-family:Arial,sans-serif;line-height:1.6">
      <h2 style="margin:0 0 12px">Verify your email</h2>
      <p style="margin:0 0 12px">Your FXCO-PILOT code is:</p>
      <div style="font-size:28px;font-weight:800;letter-spacing:4px;margin:12px 0">{code}</div>
      <p style="margin:0;color:#555">This code expires in {int(OTP_TTL_SECONDS/60)} minutes.</p>
    </div>
    """

    payload = {
        "from": RESEND_FROM,
        "to": [email],
        "subject": subject,
        "html": html,
    }

    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=12,
        )
        if r.status_code >= 200 and r.status_code < 300:
            return True, ""
        return False, f"Resend error: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return False, str(e)


# ==========================
# Tiered rate check (UPDATED)
# ==========================
def _rate_check(ip: str):
    """
    Returns: (ok: bool, retry_after_seconds: int, headers: dict)

    Tier logic:
      - paid: if _is_client_unlocked(client_id) == True
      - trial: if email is verified and within trial window
      - guest: everything else (very limited)
    Identity key:
      - use verified email when present (strong anti-spam)
      - else use client_id (localStorage UUID)
      - else use ip
    """
    now = time.time()

    client_id = _get_client_id_header() or ("ip:" + (ip or "unknown"))
    auth_email = _get_auth_email()

    is_paid = _is_client_unlocked(client_id)
    trial_active, trial_end, trial_started = _trial_status(auth_email)

    if is_paid:
        tier = "paid"
        short_limit = _env_int("PAID_LIMIT_60S", default=60, minv=1, maxv=100000)
        day_limit = _env_int("PAID_LIMIT_24H", default=2000, minv=1, maxv=10000000)
    elif trial_active:
        tier = "trial"
        short_limit = _env_int("FREE_LIMIT_60S", default=5, minv=1, maxv=100000)
        day_limit = _env_int("FREE_LIMIT_24H", default=50, minv=1, maxv=10000000)
    else:
        tier = "guest"
        short_limit = _env_int("GUEST_LIMIT_60S", default=2, minv=1, maxv=100000)
        day_limit = _env_int("GUEST_LIMIT_24H", default=10, minv=1, maxv=10000000)

    if auth_email:
        ident = f"email:{auth_email}"
    elif client_id:
        ident = f"cid:{client_id}"
    else:
        ident = f"ip:{ip or 'unknown'}"

    # init deques
    if ident not in _RATE_SHORT:
        _RATE_SHORT[ident] = deque()
    if ident not in _RATE_DAY:
        _RATE_DAY[ident] = deque()

    short_q = _RATE_SHORT[ident]
    day_q = _RATE_DAY[ident]

    # prune
    while short_q and (now - short_q[0]) >= _SHORT_WINDOW_SECONDS:
        short_q.popleft()
    while day_q and (now - day_q[0]) >= _DAY_WINDOW_SECONDS:
        day_q.popleft()

    short_remaining = max(0, short_limit - len(short_q))
    day_remaining = max(0, day_limit - len(day_q))

    # blocked
    if short_remaining <= 0 or day_remaining <= 0:
        retry_after = 1
        if short_remaining <= 0 and short_q:
            retry_after = max(retry_after, int(_SHORT_WINDOW_SECONDS - (now - short_q[0])) + 1)
        if day_remaining <= 0 and day_q:
            retry_after = max(retry_after, int(_DAY_WINDOW_SECONDS - (now - day_q[0])) + 1)

        headers = {
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit-60s": str(short_limit),
            "X-RateLimit-Remaining-60s": str(short_remaining),
            "X-RateLimit-Limit-24h": str(day_limit),
            "X-RateLimit-Remaining-24h": str(day_remaining),
            "X-FXCO-Tier": tier,
        }
        if tier == "trial" and trial_end:
            headers["X-FXCO-Trial-Ends"] = str(trial_end)
        return False, retry_after, headers

    # allow: record
    short_q.append(now)
    day_q.append(now)

    headers = {
        "X-RateLimit-Limit-60s": str(short_limit),
        "X-RateLimit-Remaining-60s": str(short_remaining - 1),
        "X-RateLimit-Limit-24h": str(day_limit),
        "X-RateLimit-Remaining-24h": str(day_remaining - 1),
        "X-FXCO-Tier": tier,
    }
    if tier == "trial" and trial_end:
        headers["X-FXCO-Trial-Ends"] = str(trial_end)
    return True, 0, headers


# ==========================
# Root + health
# ==========================
@app.route("/", methods=["GET"])
def index():
    return jsonify({"ok": True, "service": "fxco-pilot-backend", "hint": "Use /health and /api/analyze"}), 200

@app.get("/health")
def health():
    return jsonify({"ok": True}), 200


# ==========================
# Auth endpoints (Resend OTP)
# ==========================
@app.post("/api/auth/start")
def auth_start():
    """
    POST JSON: { email }
    Sends OTP to email.
    """
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not _email_ok(email):
        return jsonify({"ok": False, "error": "Invalid email."}), 400

    if not RESEND_API_KEY or not RESEND_FROM:
        return jsonify({"ok": False, "error": "Email service not configured on server."}), 500

    code = f"{int.from_bytes(os.urandom(3), 'big') % 1000000:06d}"
    exp = _now() + OTP_TTL_SECONDS
    _OTP[email] = {"code": code, "exp": exp, "attempts": 0}

    ok, err = _resend_send_otp(email, code)
    if not ok:
        return jsonify({"ok": False, "error": err or "Failed to send OTP."}), 502

    return jsonify({"ok": True, "message": "OTP sent.", "ttl_seconds": OTP_TTL_SECONDS}), 200


@app.post("/api/auth/verify")
def auth_verify():
    """
    POST JSON: { email, code }
    Verifies OTP and returns signed token.
    Also starts/returns the 14-day trial window.
    """
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    code = (data.get("code") or "").strip()

    if not _email_ok(email):
        return jsonify({"ok": False, "error": "Invalid email."}), 400
    if not re.fullmatch(r"\d{6}", code or ""):
        return jsonify({"ok": False, "error": "Invalid code format."}), 400

    row = _OTP.get(email)
    if not row:
        return jsonify({"ok": False, "error": "No OTP request found. Please request a new code."}), 400

    if _now() > float(row.get("exp") or 0):
        _OTP.pop(email, None)
        return jsonify({"ok": False, "error": "Code expired. Please request a new code."}), 400

    row["attempts"] = int(row.get("attempts") or 0) + 1
    if row["attempts"] > 8:
        _OTP.pop(email, None)
        return jsonify({"ok": False, "error": "Too many attempts. Please request a new code."}), 429

    if code != (row.get("code") or ""):
        return jsonify({"ok": False, "error": "Incorrect code."}), 400

    # success
    _OTP.pop(email, None)

    trial = _ensure_trial(email)
    trial_active, trial_end, _ = _trial_status(email)

    # token (24h)
    iat = int(_now())
    exp = iat + 24 * 60 * 60
    token = _sign_token({"email": email, "iat": iat, "exp": exp})

    if not token:
        return jsonify({"ok": False, "error": "AUTH_SIGNING_SECRET missing on server."}), 500

    return jsonify(
        {
            "ok": True,
            "token": token,
            "email": email,
            "trial_active": trial_active,
            "trial_ends": trial_end,
            "trial_days": _trial_days(),
        }
    ), 200


# ==========================
# Paystack endpoints
# ==========================
@app.get("/api/paystack/config")
def paystack_config():
    """
    REQUIRED:
      If require_payment=1/true AND secret missing => ok:false + error.
    """
    enabled = _require_payment()

    if enabled and not PAYSTACK_SECRET_KEY:
        return jsonify(
            {
                "ok": False,
                "require_payment": True,
                "error": "Paystack not configured on server (missing PAYSTACK_SECRET_KEY).",
                "currency": PAYSTACK_CURRENCY,
                "amount": _paystack_amount_minor(),
                "public_key": PAYSTACK_PUBLIC_KEY or None,
                "access_days": _access_days(),
            }
        ), 200

    return jsonify(
        {
            "ok": True,
            "require_payment": enabled,
            "currency": PAYSTACK_CURRENCY,
            "amount": _paystack_amount_minor(),
            "public_key": PAYSTACK_PUBLIC_KEY or None,
            "access_days": _access_days(),
        }
    ), 200


@app.post("/api/paystack/init")
def paystack_init():
    """
    Creates Paystack hosted checkout session.
    Expects JSON: { "email": "user@email.com" }
    Returns: { ok, authorization_url, reference }
    """
    if not _require_payment():
        return jsonify({"ok": False, "error": "Payment gate is disabled on server."}), 400

    if not PAYSTACK_SECRET_KEY:
        return jsonify({"ok": False, "error": "Missing PAYSTACK_SECRET_KEY on server."}), 500

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not _email_ok(email):
        return jsonify({"ok": False, "error": "Invalid email."}), 400

    client_id = _get_client_id_header()
    if not client_id:
        client_id = "ip:" + _client_ip()

    amount = int(_paystack_amount_minor())
    currency = PAYSTACK_CURRENCY

    callback_url = _ensure_paystack_success_param(_callback_url())

    payload = {
        "email": email,
        "amount": amount,
        "currency": currency,
        "callback_url": callback_url,
        "metadata": {
            "fxco_client_id": client_id,
            "product": "FXCO-PILOT",
            "access_days": _access_days(),
        },
    }

    try:
        r = requests.post(
            f"{PAYSTACK_BASE}/transaction/initialize",
            headers=_paystack_headers(),
            data=json.dumps(payload),
            timeout=15,
        )
        j = r.json()
        if not j.get("status"):
            return jsonify({"ok": False, "error": j.get("message") or "Paystack init failed."}), 502

        auth_url = (j.get("data") or {}).get("authorization_url")
        reference = (j.get("data") or {}).get("reference")

        if reference:
            _PENDING_REF[reference] = {"client_id": client_id, "created": _now()}

        return jsonify({"ok": True, "authorization_url": auth_url, "reference": reference}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/paystack/verify")
def paystack_verify():
    """
    Verifies a Paystack reference and unlocks access for mapped client_id.
    Query: ?reference=xxxx
    Returns: { ok, paid, reference, paid_until }
    """
    if not _require_payment():
        return jsonify({"ok": False, "error": "Payment gate is disabled on server."}), 400

    if not PAYSTACK_SECRET_KEY:
        return jsonify({"ok": False, "error": "Missing PAYSTACK_SECRET_KEY on server."}), 500

    reference = (request.args.get("reference") or "").strip()
    if not reference:
        return jsonify({"ok": False, "error": "Missing reference."}), 400

    try:
        r = requests.get(f"{PAYSTACK_BASE}/transaction/verify/{reference}", headers=_paystack_headers(), timeout=15)
        j = r.json()

        if not j.get("status"):
            return jsonify({"ok": False, "error": j.get("message") or "Paystack verify failed."}), 502

        data = j.get("data") or {}
        status = (data.get("status") or "").lower()
        currency = (data.get("currency") or "").upper()
        amount = int(data.get("amount") or 0)

        expected_amount = int(_paystack_amount_minor())
        expected_currency = PAYSTACK_CURRENCY

        paid = (status == "success") and (currency == expected_currency) and (amount >= expected_amount)

        meta = data.get("metadata") or {}
        client_id = (meta.get("fxco_client_id") or "").strip()

        if not client_id:
            pending = _PENDING_REF.get(reference) or {}
            client_id = pending.get("client_id", "")

        if paid and client_id:
            paid_until = _now() + _access_days() * 24 * 60 * 60
            _ACCESS[client_id] = {"paid_until": paid_until, "last_ref": reference, "updated": _now()}
            _PENDING_REF.pop(reference, None)
            return jsonify({"ok": True, "paid": True, "reference": reference, "paid_until": int(paid_until)}), 200

        return jsonify({"ok": True, "paid": False, "reference": reference}), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ==========================
# FX helpers
# ==========================
def _norm_symbol(s: str) -> str:
    return (s or "").upper().replace("/", "").replace("-", "").replace(" ", "").strip()

def detect_symbol_from_signal(signal_text: str, pair_type: str) -> str:
    txt = (signal_text or "").upper()

    m = re.search(r"\b([A-Z]{3,5})\s*/\s*([A-Z]{3,5})\b", txt)
    if m:
        return _norm_symbol(m.group(1) + m.group(2))

    m = re.search(r"\b([A-Z]{6})\b", txt)
    if m:
        return _norm_symbol(m.group(1))

    m = re.search(r"\b(XAUUSD|XAGUSD|BTCUSD|ETHUSD|SOLUSD|XRPUSD)\b", txt)
    if m:
        return _norm_symbol(m.group(1))

    pt = (pair_type or "").lower()
    if pt == "gold":
        return "XAUUSD"
    if pt == "crypto":
        return "BTCUSD"
    return "EURUSD"

def _to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def _parse_targets(val):
    if val is None:
        return []
    if isinstance(val, list):
        out = []
        for x in val:
            fx = _to_float(x)
            if fx is not None:
                out.append(fx)
        return out
    s = str(val)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            pass
    return out

def calculate_rr(entry, stop, targets):
    entry_f = _to_float(entry)
    stop_f = _to_float(stop)
    tps = _parse_targets(targets)
    if entry_f is None or stop_f is None or not tps:
        return None
    risk = abs(entry_f - stop_f)
    reward = abs(tps[0] - entry_f)
    if risk <= 0:
        return None
    return round(reward / risk, 2)


# ==========================
# Twelve Data calls (cached)
# ==========================
def td_price(symbol: str):
    if not TWELVE_DATA_API_KEY:
        return {"ok": False, "error": "Missing TWELVE_DATA_API_KEY (live data disabled)."}

    symbol = _norm_symbol(symbol or "EURUSD")
    cache_key = f"td_price::{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        r = requests.get(
            f"{TD_BASE}/price",
            params={"symbol": symbol, "apikey": TWELVE_DATA_API_KEY},
            timeout=10,
        )
        data = r.json()
        if data.get("status") == "error":
            out = {"ok": False, "error": data.get("message", "Twelve Data error")}
            _cache_set(cache_key, out, ttl=15)
            return out

        p = float(data["price"])
        out = {"ok": True, "symbol": symbol, "price": p, "source": "twelvedata"}
        _cache_set(cache_key, out, ttl=15)
        return out
    except Exception as e:
        out = {"ok": False, "error": str(e)}
        _cache_set(cache_key, out, ttl=15)
        return out


def td_candles(symbol: str, interval: str = "5min", limit: int = 120):
    if not TWELVE_DATA_API_KEY:
        return {"ok": False, "error": "Missing TWELVE_DATA_API_KEY (live data disabled)."}

    symbol = _norm_symbol(symbol or "EURUSD")
    interval = (interval or "5min").strip()
    limit = int(limit or 120)

    cache_key = f"td_candles::{symbol}::{interval}::{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        r = requests.get(
            f"{TD_BASE}/time_series",
            params={
                "symbol": symbol,
                "interval": interval,
                "outputsize": limit,
                "apikey": TWELVE_DATA_API_KEY,
            },
            timeout=12,
        )
        data = r.json()
        if data.get("status") == "error":
            out = {"ok": False, "error": data.get("message", "Twelve Data error")}
            _cache_set(cache_key, out, ttl=30)
            return out

        values = data.get("values") or []
        out = {"ok": True, "symbol": symbol, "interval": interval, "values": values, "source": "twelvedata"}
        _cache_set(cache_key, out, ttl=30)
        return out
    except Exception as e:
        out = {"ok": False, "error": str(e)}
        _cache_set(cache_key, out, ttl=30)
        return out


def structure_from_candles(values):
    if not values or len(values) < 40:
        return {"structure": "unclear", "broken": False, "details": "Not enough candle data."}

    vals = list(reversed(values))  # oldest -> newest

    try:
        closes = [float(v["close"]) for v in vals]
        highs = [float(v["high"]) for v in vals]
        lows = [float(v["low"]) for v in vals]
    except Exception:
        return {"structure": "unclear", "broken": False, "details": "Candle parse error."}

    last = closes[-1]
    prev = closes[-25]
    trend = "bullish" if last > prev else "bearish" if last < prev else "unclear"

    recent_low = min(lows[-15:])
    prior_low = min(lows[-40:-15])
    recent_high = max(highs[-15:])
    prior_high = max(highs[-40:-15])

    if trend == "bullish":
        if recent_low > prior_low:
            return {"structure": "bullish", "broken": False, "details": "Higher low detected."}
        return {"structure": "unclear", "broken": False, "details": "Bullish trend but HL not confirmed."}

    if trend == "bearish":
        if recent_high < prior_high:
            return {"structure": "bearish", "broken": False, "details": "Lower high detected."}
        return {"structure": "unclear", "broken": False, "details": "Bearish trend but LH not confirmed."}

    return {"structure": "unclear", "broken": False, "details": "Trend unclear."}


def compute_confidence(analysis):
    score = 0
    mc = analysis.get("market_context", {}) or {}
    sc = analysis.get("signal_check", {}) or {}

    struct = (mc.get("structure") or "").lower()
    if "bullish" in struct or "bearish" in struct:
        score += 30
    elif struct:
        score += 15

    rr = sc.get("rr")
    if isinstance(rr, (int, float)):
        if rr >= 2.5:
            score += 25
        elif rr >= 2.0:
            score += 20
        elif rr >= 1.5:
            score += 12
        elif rr >= 1.2:
            score += 6

    liquidity = (mc.get("liquidity") or "").lower()
    if any(x in liquidity for x in ["liquidity", "sweep", "grab", "equal"]):
        score += 20
    elif liquidity:
        score += 10

    momentum = (mc.get("momentum") or "").lower()
    if "strong" in momentum:
        score += 15
    elif momentum:
        score += 8

    if sc.get("entry") and sc.get("stop_loss") and sc.get("targets"):
        score += 10
    elif sc.get("entry") and sc.get("stop_loss"):
        score += 6

    return max(0, min(100, score))


def build_why_this_trade(analysis):
    mc = analysis.get("market_context", {}) or {}
    sc = analysis.get("signal_check", {}) or {}

    reasons = []
    if mc.get("structure"):
        reasons.append(f"Structure context: {mc.get('structure')}.")
    if mc.get("liquidity"):
        reasons.append(f"Liquidity logic: {mc.get('liquidity')}.")
    if mc.get("momentum"):
        reasons.append(f"Momentum: {mc.get('momentum')}.")
    rr = sc.get("rr")
    if rr is not None:
        reasons.append(f"Risk/Reward (TP1): RR ≈ {rr}.")
    if sc.get("entry") and sc.get("stop_loss"):
        reasons.append("Defined entry and stop loss reduces ambiguity.")
    return reasons[:5]


def build_invalidation_warnings(analysis, live_snapshot=None):
    warnings = []
    sc = analysis.get("signal_check", {}) or {}
    bias = (analysis.get("bias") or "unclear").lower()
    struct = (analysis.get("market_context", {}) or {}).get("structure") or "unclear"

    if not sc.get("entry") or not sc.get("stop_loss") or not sc.get("targets"):
        warnings.append("Missing key levels (entry / SL / targets).")

    rr = sc.get("rr")
    if isinstance(rr, (int, float)) and rr < 1.2:
        warnings.append(f"Low RR (≈ {rr}). Consider skipping or improving RR.")

    if live_snapshot and live_snapshot.get("ok"):
        price = live_snapshot.get("price")
        sl = _to_float(sc.get("stop_loss"))
        if sl is not None:
            if "long" in bias and price <= sl:
                warnings.append("Live price has hit SL => invalidated.")
            if "short" in bias and price >= sl:
                warnings.append("Live price has hit SL => invalidated.")

    if ("long" in bias and "bearish" in struct.lower()) or ("short" in bias and "bullish" in struct.lower()):
        warnings.append("Structure is against your bias (structure broken).")

    return warnings[:8]


def _pick_first(data: dict, keys: list[str]) -> str:
    for k in keys:
        v = data.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def _get_payload_fields():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        pair_type = _pick_first(data, ["pair_type", "pairType"])
        timeframe = _pick_first(data, ["timeframe", "timeframe_mode", "timeframeMode"])
        signal_text = _pick_first(data, ["signal_input", "signal", "signalText"])
        return pair_type, timeframe, signal_text

    form = request.form or {}
    pair_type = _pick_first(form, ["pair_type", "pairType"])
    timeframe = _pick_first(form, ["timeframe", "timeframe_mode", "timeframeMode"])
    signal_text = _pick_first(form, ["signal_input", "signal", "signalText"])
    return pair_type, timeframe, signal_text


# ==========================
# Analyze endpoint (trial OR paid)
# ==========================
@app.route("/api/analyze", methods=["POST"])
def analyze():
    ip = _client_ip()

    ok, retry_after, rl_headers = _rate_check(ip)
    if not ok:
        resp = jsonify(
            {
                "error": "Too many requests. Please slow down.",
                "retry_after_seconds": retry_after,
            }
        )
        resp.status_code = 429
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp

    # OPTIONAL: test mode
    if request.args.get("test") == "1":
        resp = jsonify({"ok": True, "mode": "rate_limit_test", "note": "No OpenAI call made."})
        resp.status_code = 200
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp

    client_id = _get_client_id_header() or ("ip:" + ip)
    auth_email = _get_auth_email()
    trial_active, trial_end, trial_started = _trial_status(auth_email)

    # Gate logic:
    # If payment required: allow if PAID OR TRIAL_ACTIVE. Otherwise 402.
    if _require_payment():
        if not _is_client_unlocked(client_id) and not trial_active:
            msg = "Payment required. Trial ended or not verified. Please unlock via Paystack."
            # Give a more helpful hint if they never verified email
            if not auth_email:
                msg = "Verify your email to start free trial, or unlock via Paystack."
            resp = jsonify({"error": msg})
            resp.status_code = 402
            for k, v in rl_headers.items():
                resp.headers[k] = v
            return resp

    # Parse payload
    pair_type, timeframe, signal_text = _get_payload_fields()
    missing = []
    if not pair_type:
        missing.append("pair_type")
    if not timeframe:
        missing.append("timeframe")
    if not signal_text:
        missing.append("signal_input")

    if missing:
        resp = jsonify(
            {
                "error": "Missing required fields.",
                "missing": missing,
                "received_content_type": request.headers.get("Content-Type", ""),
            }
        )
        resp.status_code = 400
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp

    oa = _get_openai_client()
    if oa is None:
        resp = jsonify({"error": "Missing OPENAI_API_KEY on server."})
        resp.status_code = 500
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp

    img_base64 = None
    file = request.files.get("chart_image")
    if file and file.filename:
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    symbol = detect_symbol_from_signal(signal_text, pair_type)
    live_snapshot = td_price(symbol)
    candles_snapshot = td_candles(symbol, interval="5min", limit=120)

    struct_info = {"structure": "unclear", "broken": False, "details": ""}
    if candles_snapshot.get("ok") and candles_snapshot.get("values"):
        struct_info = structure_from_candles(candles_snapshot["values"])

    live_context = "Live data unavailable."
    if live_snapshot.get("ok"):
        live_context = f"Live price: {live_snapshot.get('price')} ({symbol})"
    else:
        live_context = f"Live data error: {live_snapshot.get('error', 'unknown')}"

    base_prompt = f"""
You are FX CO-PILOT — an institutional-grade trade validation engine.

User Context:
- Pair type: {pair_type}
- Timeframe mode: {timeframe}
- Raw signal:
\"\"\"{signal_text}\"\"\"

Live Market:
- Symbol: {symbol}
- {live_context}
- 5min Structure (heuristic from candles): {struct_info.get('structure')} ({struct_info.get('details')})

Return ONLY valid JSON that matches this schema (no markdown):

{{
  "bias": "Long|Short|Neutral|Unclear",
  "strength": 0,
  "clarity": 0,
  "signal_check": {{
    "direction": "Long|Short|Neutral|Unclear",
    "entry": "number or a single price",
    "stop_loss": "number",
    "targets": [number]
  }},
  "market_context": {{
    "structure": "string",
    "liquidity": "string",
    "momentum": "string",
    "timeframe_alignment": "string"
  }},
  "decision": "TAKE TRADE|NEUTRAL|AVOID TRADE",
  "verdict": "string",
  "guidance": ["string","string","string"]
}}
Rules:
- Output MUST be raw JSON only.
- entry/stop_loss/targets MUST be numeric-like.
- If uncertain, choose NEUTRAL.
"""

    messages = [
        {"role": "system", "content": "You are FX Co-Pilot. Output ONLY JSON."},
        {"role": "user", "content": base_prompt},
    ]

    if img_base64:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the user's chart screenshot. Use it to refine structure/liquidity/trend."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                ],
            }
        )

    try:
        completion = oa.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.2,
        )

        raw = completion.choices[0].message.content or ""
        analysis_obj = json.loads(raw)

        sc = analysis_obj.get("signal_check") or {}
        mc = analysis_obj.get("market_context") or {}

        rr = calculate_rr(sc.get("entry"), sc.get("stop_loss"), sc.get("targets"))

        analysis = {
            "bias": analysis_obj.get("bias") or "Unclear",
            "strength": analysis_obj.get("strength") or 0,
            "clarity": analysis_obj.get("clarity") or 0,
            "signal_check": {
                "direction": sc.get("direction") or "Unclear",
                "entry": _to_float(sc.get("entry")),
                "stop_loss": _to_float(sc.get("stop_loss")),
                "targets": _parse_targets(sc.get("targets")),
                "rr": rr,
            },
            "market_context": {
                "structure": (mc.get("structure") or "") + f" | Live(5m): {struct_info.get('structure')}",
                "liquidity": mc.get("liquidity") or "",
                "momentum": mc.get("momentum") or "",
                "timeframe_alignment": mc.get("timeframe_alignment") or "",
            },
            "decision": analysis_obj.get("decision") or "NEUTRAL",
            "verdict": analysis_obj.get("verdict") or "",
            "guidance": analysis_obj.get("guidance") or [],
            "live_snapshot": live_snapshot,
        }

        d = str(analysis.get("decision") or "NEUTRAL").upper()
        if "TAKE" in d:
            analysis["decision"] = "TAKE TRADE"
        elif "AVOID" in d:
            analysis["decision"] = "AVOID TRADE"
        else:
            analysis["decision"] = "NEUTRAL"

        analysis["confidence"] = compute_confidence(analysis)
        analysis["why_this_trade"] = build_why_this_trade(analysis)
        analysis["invalidation_warnings"] = build_invalidation_warnings(analysis, live_snapshot=live_snapshot)

        resp = jsonify({"blocked": False, "analysis": analysis, "mode": "twelvedata_live"})
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp

    except json.JSONDecodeError:
        resp = jsonify({"error": "Model did not return valid JSON."})
        resp.status_code = 502
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp
    except Exception as e:
        msg = str(e)
        status = 500
        if "insufficient_quota" in msg or "quota" in msg.lower():
            status = 402
        resp = jsonify({"error": msg})
        resp.status_code = status
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
