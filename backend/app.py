import base64
import json
import os
import re
import time
from collections import deque
from random import randint

import requests
from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)

# ============================================================
# ENV (set these on Render/your host)
# ============================================================
# OpenAI:
#   OPENAI_API_KEY=...
#
# Twelve Data:
#   TWELVE_DATA_API_KEY=...
#
# Paystack:
#   PAYSTACK_SECRET_KEY=sk_live_or_test_xxx
#   PAYSTACK_PUBLIC_KEY=pk_live_or_test_xxx   # optional (hosted checkout doesn't need it)
#
# Pricing:
#   PAYSTACK_AMOUNT_NGN=10000
#   PAYSTACK_CURRENCY=NGN
#   PAYSTACK_REQUIRE_PAYMENT=1                # 1/true to gate analyze after trial ends
#   PAYSTACK_CALLBACK_URL=https://YOUR-FRONTEND-DOMAIN/
#   PAYSTACK_ACCESS_DAYS=30
#
# Trial (Resend):
#   RESEND_API_KEY=...
#   RESEND_FROM_EMAIL="FXCO-PILOT <no-reply@yourdomain.com>"
#   TRIAL_DAYS=14
#   TRIAL_OTP_TTL_SECONDS=600                 # 10 minutes default
#
# Rate limit overrides (optional):
#   FREE_LIMIT_60S=5
#   FREE_LIMIT_24H=50
#   PAID_LIMIT_60S=60
#   PAID_LIMIT_24H=2000
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
# OpenAI client (lazy init)
# ==========================
def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# ==========================
# Twelve Data
# ==========================
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TD_BASE = "https://api.twelvedata.com"


# ==========================
# Paystack config + state (in-memory)
# ==========================
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY", "").strip()
PAYSTACK_PUBLIC_KEY = os.getenv("PAYSTACK_PUBLIC_KEY", "").strip()
PAYSTACK_CURRENCY = (os.getenv("PAYSTACK_CURRENCY", "NGN") or "NGN").strip().upper()
PAYSTACK_BASE = "https://api.paystack.co"

# pending reference -> client_id (mapping after verify)
_PENDING_REF = {}  # ref -> {"client_id": str, "created": epoch}
# access client_id -> paid_until_epoch
_ACCESS = {}  # client_id -> {"paid_until": epoch, "last_ref": str, "updated": epoch}


def _paystack_amount_minor():
    # ₦10,000/month => 10000 NGN major => 1,000,000 kobo minor
    try:
        major = int((os.getenv("PAYSTACK_AMOUNT_NGN", "10000") or "10000").strip())
    except Exception:
        major = 10000
    return max(0, major) * 100


def _require_payment():
    v = (os.getenv("PAYSTACK_REQUIRE_PAYMENT", "1") or "1").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _access_days():
    try:
        return max(1, int((os.getenv("PAYSTACK_ACCESS_DAYS", "30") or "30").strip()))
    except Exception:
        return 30


def _now():
    return time.time()


def _cleanup_payment_state():
    now = _now()
    # pending refs expire after 24h
    for ref in list(_PENDING_REF.keys()):
        if now - _PENDING_REF[ref].get("created", now) > 24 * 60 * 60:
            _PENDING_REF.pop(ref, None)
    # access entries expire after paid_until + 7 days grace
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
    # If payment not required => OK
    # If required => need secret key
    if not _require_payment():
        return True
    return bool(PAYSTACK_SECRET_KEY)


def _callback_url():
    cb = (os.getenv("PAYSTACK_CALLBACK_URL", "") or "").strip()
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


# ==========================
# Trial via Resend (in-memory)
# ==========================
RESEND_API_KEY = (os.getenv("RESEND_API_KEY", "") or "").strip()
RESEND_FROM_EMAIL = (os.getenv("RESEND_FROM_EMAIL", "") or "").strip()


def _trial_days():
    try:
        return max(1, int((os.getenv("TRIAL_DAYS", "14") or "14").strip()))
    except Exception:
        return 14


def _otp_ttl():
    try:
        return max(60, int((os.getenv("TRIAL_OTP_TTL_SECONDS", "600") or "600").strip()))
    except Exception:
        return 600


# email -> { trial_until, started, verified }
_TRIAL = {}
# email -> { code, expires, created }
_TRIAL_OTP = {}
# client_id -> verified email
_CLIENT_EMAIL = {}


def _cleanup_trial_state():
    now = _now()
    # OTP expiry cleanup
    for email in list(_TRIAL_OTP.keys()):
        if now > float((_TRIAL_OTP[email] or {}).get("expires", 0)):
            _TRIAL_OTP.pop(email, None)
    # trial cleanup (optional) - keep records to prevent re-trials
    # We intentionally DO NOT delete expired trials, so the same email can't restart.


def _email_ok(email: str) -> bool:
    if not email:
        return False
    email = email.strip()
    if len(email) > 254:
        return False
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email))


def _send_resend_email(to_email: str, subject: str, html: str):
    if not RESEND_API_KEY:
        raise RuntimeError("Missing RESEND_API_KEY on server.")
    if not RESEND_FROM_EMAIL:
        raise RuntimeError("Missing RESEND_FROM_EMAIL on server.")

    r = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
        data=json.dumps(
            {
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": subject,
                "html": html,
            }
        ),
        timeout=15,
    )
    # Resend returns JSON with id on success. If error, it also returns JSON.
    j = r.json() if r.content else {}
    if r.status_code >= 400:
        msg = (j.get("message") or j.get("error") or f"Resend error (HTTP {r.status_code})")
        raise RuntimeError(msg)
    return j


def _trial_active_for_email(email: str) -> bool:
    _cleanup_trial_state()
    if not email:
        return False
    row = _TRIAL.get(email)
    if not row:
        return False
    return _now() < float(row.get("trial_until", 0))


def _trial_status(email: str):
    row = _TRIAL.get(email) or {}
    return {
        "email": email or None,
        "verified": bool(row.get("verified")),
        "trial_until": int(row.get("trial_until") or 0) if row.get("trial_until") else 0,
        "trial_active": _trial_active_for_email(email),
        "trial_days": _trial_days(),
    }


# ==========================
# Client identifiers
# ==========================
def _client_ip():
    """
    Try to get the real client IP behind proxies.
    X-Forwarded-For is usually: "client, proxy1, proxy2"
    """
    xff = request.headers.get("X-Forwarded-For", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _get_client_id_header():
    cid = (request.headers.get("X-FXCO-Client") or "").strip()
    if cid:
        return cid[:128]
    return ""


def _client_id_fallback(ip: str) -> str:
    # stable fallback if frontend doesn't send X-FXCO-Client
    return "ip:" + (ip or "unknown")


def _client_verified_email(client_id: str) -> str:
    if not client_id:
        return ""
    return (_CLIENT_EMAIL.get(client_id) or "").strip().lower()


def _is_client_unlocked(client_id: str) -> bool:
    _cleanup_payment_state()
    if not _require_payment():
        return True
    if not client_id:
        return False
    row = _ACCESS.get(client_id)
    if not row:
        return False
    return _now() < float(row.get("paid_until", 0))


# ==========================
# Rate limiting (in-memory)
#   - FREE + TRIAL: same limits
#   - PAID: higher limits
# Key: verified email -> else client_id -> else ip
# Applied ONLY to /api/analyze
# ==========================
_RATE_SHORT = {}  # key -> deque[timestamps]
_RATE_DAY = {}  # key -> deque[timestamps]


def _env_int(name: str, default: int, minv: int = 1, maxv: int = 10_000_000) -> int:
    try:
        v = int((os.getenv(name, str(default)) or str(default)).strip())
        v = max(minv, min(maxv, v))
        return v
    except Exception:
        return default


def _limits_for_user(is_paid: bool):
    free_60 = _env_int("FREE_LIMIT_60S", 5, minv=1, maxv=1000000)
    free_24 = _env_int("FREE_LIMIT_24H", 50, minv=1, maxv=10000000)

    paid_60 = _env_int("PAID_LIMIT_60S", 60, minv=1, maxv=1000000)
    paid_24 = _env_int("PAID_LIMIT_24H", 2000, minv=1, maxv=10000000)

    # Trial uses FREE limits (per your request)
    if is_paid:
        return 60, paid_60, 24 * 60 * 60, paid_24

    return 60, free_60, 24 * 60 * 60, free_24


def _rate_check(ip: str, client_id: str):
    """
    Returns: (ok: bool, retry_after_seconds: int, headers: dict)

    - Detects PAID status via _is_client_unlocked(client_id)
    - Applies limits automatically (no frontend change)
    - Uses key priority: verified_email -> client_id -> ip
    """
    now = time.time()

    cid = (client_id or "").strip()
    if not cid:
        cid = _client_id_fallback(ip)

    verified_email = _client_verified_email(cid)
    key = verified_email or cid or (ip or "unknown")

    is_paid = _is_client_unlocked(cid)

    short_window, short_limit, day_window, day_limit = _limits_for_user(is_paid)

    if key not in _RATE_SHORT:
        _RATE_SHORT[key] = deque()
    if key not in _RATE_DAY:
        _RATE_DAY[key] = deque()

    short_q = _RATE_SHORT[key]
    day_q = _RATE_DAY[key]

    # prune old
    while short_q and (now - short_q[0]) >= short_window:
        short_q.popleft()
    while day_q and (now - day_q[0]) >= day_window:
        day_q.popleft()

    short_remaining = max(0, short_limit - len(short_q))
    day_remaining = max(0, day_limit - len(day_q))

    # blocked
    if short_remaining <= 0 or day_remaining <= 0:
        retry_after = 1
        if short_remaining <= 0 and short_q:
            retry_after = max(retry_after, int(short_window - (now - short_q[0])) + 1)
        if day_remaining <= 0 and day_q:
            retry_after = max(retry_after, int(day_window - (now - day_q[0])) + 1)

        headers = {
            "Retry-After": str(retry_after),
            "X-RateLimit-Key": "email" if verified_email else ("client" if cid else "ip"),
            "X-RateLimit-Plan": "paid" if is_paid else "free",
            "X-RateLimit-Limit-60s": str(short_limit),
            "X-RateLimit-Remaining-60s": str(short_remaining),
            "X-RateLimit-Limit-24h": str(day_limit),
            "X-RateLimit-Remaining-24h": str(day_remaining),
        }
        return False, retry_after, headers

    # allow: record
    short_q.append(now)
    day_q.append(now)

    headers = {
        "X-RateLimit-Key": "email" if verified_email else ("client" if cid else "ip"),
        "X-RateLimit-Plan": "paid" if is_paid else "free",
        "X-RateLimit-Limit-60s": str(short_limit),
        "X-RateLimit-Remaining-60s": str(short_remaining - 1),
        "X-RateLimit-Limit-24h": str(day_limit),
        "X-RateLimit-Remaining-24h": str(day_remaining - 1),
    }
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
# Trial endpoints (Resend)
# ==========================
@app.post("/api/trial/start")
def trial_start():
    """
    Starts email verification (OTP). Does NOT create/extend a trial if the email already had one.
    Expects JSON: { "email": "user@email.com" }
    """
    _cleanup_trial_state()

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not _email_ok(email):
        return jsonify({"ok": False, "error": "Invalid email."}), 400

    # If trial already exists (active or expired), do not allow restarting.
    if email in _TRIAL:
        s = _trial_status(email)
        return jsonify(
            {
                "ok": True,
                "already_started": True,
                "message": "Trial already exists for this email.",
                "trial": s,
            }
        ), 200

    # Generate OTP
    code = f"{randint(0, 999999):06d}"
    expires = _now() + _otp_ttl()
    _TRIAL_OTP[email] = {"code": code, "expires": expires, "created": _now()}

    # Send via Resend
    try:
        subject = "Your FXCO-PILOT verification code"
        html = f"""
        <div style="font-family:Arial,sans-serif;line-height:1.6">
          <h2>FXCO-PILOT Email Verification</h2>
          <p>Your code is:</p>
          <div style="font-size:28px;font-weight:800;letter-spacing:6px;margin:12px 0">{code}</div>
          <p>This code expires in {int(_otp_ttl() / 60)} minutes.</p>
          <p style="opacity:0.8">If you didn’t request this, you can ignore this email.</p>
        </div>
        """
        _send_resend_email(email, subject, html)
    except Exception as e:
        # cleanup otp if sending failed
        _TRIAL_OTP.pop(email, None)
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "sent": True, "email": email}), 200


@app.post("/api/trial/verify")
def trial_verify():
    """
    Verifies OTP and activates 14-day trial for that email (one-time only).
    Also binds email -> client_id (X-FXCO-Client) so /api/analyze doesn't need new headers.
    Expects JSON: { "email": "...", "code": "123456" }
    """
    _cleanup_trial_state()

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    code = (data.get("code") or "").strip()

    if not _email_ok(email):
        return jsonify({"ok": False, "error": "Invalid email."}), 400
    if not re.fullmatch(r"\d{6}", code or ""):
        return jsonify({"ok": False, "error": "Invalid code format."}), 400

    # If trial already exists, don't extend. But still bind email to client_id.
    if email in _TRIAL:
        client_id = _get_client_id_header()
        if client_id:
            _CLIENT_EMAIL[client_id] = email
        return jsonify({"ok": True, "verified": True, "already_started": True, "trial": _trial_status(email)}), 200

    row = _TRIAL_OTP.get(email) or {}
    if not row:
        return jsonify({"ok": False, "error": "No pending code for this email. Start again."}), 400
    if _now() > float(row.get("expires", 0)):
        _TRIAL_OTP.pop(email, None)
        return jsonify({"ok": False, "error": "Code expired. Start again."}), 400
    if str(row.get("code") or "") != code:
        return jsonify({"ok": False, "error": "Incorrect code."}), 400

    # Activate trial (one-time)
    days = _trial_days()
    trial_until = _now() + days * 24 * 60 * 60
    _TRIAL[email] = {"trial_until": trial_until, "started": _now(), "verified": True}

    # Bind verified email to client_id so /api/analyze can identify them automatically
    client_id = _get_client_id_header()
    if client_id:
        _CLIENT_EMAIL[client_id] = email

    # consume otp
    _TRIAL_OTP.pop(email, None)

    return jsonify({"ok": True, "verified": True, "trial": _trial_status(email)}), 200


@app.get("/api/trial/status")
def trial_status():
    """
    Optional helper for UI: shows trial status for current client_id (if verified).
    """
    ip = _client_ip()
    client_id = _get_client_id_header() or _client_id_fallback(ip)
    email = _client_verified_email(client_id)
    return jsonify({"ok": True, "client_id": client_id, "trial": _trial_status(email)}), 200


# ==========================
# Paystack endpoints
# ==========================
@app.get("/api/paystack/config")
def paystack_config():
    """
    REQUIRED:
      If require_payment=1 AND secret missing => ok:false + error field.
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
                "trial_days": _trial_days(),
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
            "trial_days": _trial_days(),
        }
    ), 200


@app.post("/api/paystack/init")
def paystack_init():
    """
    Creates a Paystack hosted checkout session.
    Expects JSON: { "email": "user@email.com" }
    Returns: { ok, authorization_url, reference }

    Callback uses PAYSTACK_CALLBACK_URL and ensures paystack=success exists.
    """
    if not _require_payment():
        return jsonify({"ok": False, "error": "Payment gate is disabled on server."}), 400

    if not PAYSTACK_SECRET_KEY:
        return jsonify({"ok": False, "error": "Missing PAYSTACK_SECRET_KEY on server."}), 500

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not _email_ok(email):
        return jsonify({"ok": False, "error": "Invalid email."}), 400

    ip = _client_ip()
    client_id = _get_client_id_header() or _client_id_fallback(ip)

    amount = _paystack_amount_minor()
    currency = PAYSTACK_CURRENCY

    callback_url = _ensure_paystack_success_param(_callback_url())

    payload = {
        "email": email,
        "amount": int(amount),
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
    Verifies a Paystack reference and unlocks access for the mapped client_id for N days.
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
            days = _access_days()
            paid_until = _now() + days * 24 * 60 * 60
            _ACCESS[client_id] = {"paid_until": paid_until, "last_ref": reference, "updated": _now()}
            _PENDING_REF.pop(reference, None)
            return jsonify({"ok": True, "paid": True, "reference": reference, "paid_until": int(paid_until)}), 200

        return jsonify({"ok": True, "paid": False, "reference": reference}), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ==========================
# Helpers (existing)
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
# Analyze endpoint (Trial + Paystack gate)
# ==========================
@app.route("/api/analyze", methods=["POST"])
def analyze():
    ip = _client_ip()
    client_id = _get_client_id_header() or _client_id_fallback(ip)

    # Rate limit ONLY this endpoint (auto paid/free)
    ok, retry_after, rl_headers = _rate_check(ip, client_id)
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

    # OPTIONAL: test mode BEFORE anything expensive
    if request.args.get("test") == "1":
        resp = jsonify({"ok": True, "mode": "rate_limit_test", "note": "No OpenAI call made."})
        resp.status_code = 200
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp

    # Trial + Paystack gate
    # If PAYSTACK_REQUIRE_PAYMENT is ON:
    #   allow if trial is active OR client is paid
    if _require_payment():
        email = _client_verified_email(client_id)
        trial_ok = _trial_active_for_email(email)
        paid_ok = _is_client_unlocked(client_id)

        if not trial_ok and not paid_ok:
            resp = jsonify(
                {
                    "error": "Access blocked. Trial ended or not verified. Please pay with Paystack to continue.",
                    "trial": _trial_status(email),
                    "paid": False,
                }
            )
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
                "expected_any_of": {
                    "pair_type": ["pair_type", "pairType"],
                    "timeframe": ["timeframe", "timeframe_mode", "timeframeMode"],
                    "signal_input": ["signal_input", "signal", "signalText"],
                },
                "received_content_type": request.headers.get("Content-Type", ""),
            }
        )
        resp.status_code = 400
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp

    # OpenAI client check
    oa = _get_openai_client()
    if oa is None:
        resp = jsonify({"error": "Missing OPENAI_API_KEY on server."})
        resp.status_code = 500
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp

    # Optional image (FormData only)
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
