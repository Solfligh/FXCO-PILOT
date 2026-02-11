import base64
import hashlib
import hmac
import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import requests
from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)

# ============================================================
# ENV
# ============================================================
# OpenAI: OPENAI_API_KEY
# TwelveData: TWELVE_DATA_API_KEY
#
# Paystack:
#   PAYSTACK_SECRET_KEY
#   PAYSTACK_PUBLIC_KEY (optional)
#   PAYSTACK_CALLBACK_URL
#
# Payment gate toggle:
#   PAYSTACK_REQUIRE_PAYMENT=1  OR  REQUIRE_PAYMENT=true
#
# Pricing:
#   PAYSTACK_AMOUNT_NGN=10000   (NGN major) OR PAYSTACK_AMOUNT=1000000 (minor units)
#   PAYSTACK_CURRENCY=NGN
#   PAYSTACK_ACCESS_DAYS=30
#
# Supabase (for OTP + trials + paid access persistence):
#   SUPABASE_URL
#   SUPABASE_SERVICE_ROLE_KEY
#
# Resend:
#   RESEND_API_KEY
#   RESEND_FROM_EMAIL
#
# Trial:
#   TRIAL_DAYS=14
#
# Auth signing:
#   AUTH_SIGNING_SECRET=long_random_string
#
# Rate limits:
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
# Time helpers
# ==========================
def _now() -> float:
    return time.time()


def _utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _dt_to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(dt_str: str) -> Optional[datetime]:
    try:
        # Supabase returns ISO timestamps; Python can parse most of them
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


# ==========================
# Request identity
# ==========================
def _client_ip():
    xff = request.headers.get("X-Forwarded-For", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _get_client_id_header():
    cid = (request.headers.get("X-FXCO-Client") or "").strip()
    if cid:
        return cid[:128]
    return ""


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
# Supabase helpers (PostgREST)
# ==========================
SUPABASE_URL = (os.getenv("SUPABASE_URL", "") or "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()


def _sb_ok() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _sb_select_one(table: str, filters: dict) -> Optional[dict]:
    """
    Select first row matching filters, or None.
    """
    if not _sb_ok():
        return None

    params = {k: f"eq.{v}" for k, v in filters.items()}
    # limit=1
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_sb_headers(),
            params={**params, "select": "*", "limit": "1"},
            timeout=12,
        )
        if r.status_code >= 400:
            return None
        rows = r.json()
        if isinstance(rows, list) and rows:
            return rows[0]
        return None
    except Exception:
        return None


def _sb_upsert(table: str, row: dict, conflict: str) -> bool:
    """
    Upsert by conflict column.
    """
    if not _sb_ok():
        return False
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={**_sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
            params={"on_conflict": conflict},
            data=json.dumps(row),
            timeout=12,
        )
        return 200 <= r.status_code < 300
    except Exception:
        return False


def _sb_update(table: str, filters: dict, patch: dict) -> bool:
    if not _sb_ok():
        return False
    params = {k: f"eq.{v}" for k, v in filters.items()}
    try:
        r = requests.patch(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={**_sb_headers(), "Prefer": "return=minimal"},
            params=params,
            data=json.dumps(patch),
            timeout=12,
        )
        return 200 <= r.status_code < 300
    except Exception:
        return False


# ==========================
# Resend helpers
# ==========================
RESEND_API_KEY = (os.getenv("RESEND_API_KEY", "") or "").strip()
RESEND_FROM_EMAIL = (os.getenv("RESEND_FROM_EMAIL", "") or "").strip()


def _resend_ok() -> bool:
    return bool(RESEND_API_KEY and RESEND_FROM_EMAIL)


def _send_email_resend(to_email: str, subject: str, html: str) -> Tuple[bool, str]:
    if not _resend_ok():
        return False, "Resend not configured (missing RESEND_API_KEY or RESEND_FROM_EMAIL)."
    try:
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
            timeout=12,
        )
        if r.status_code >= 400:
            try:
                return False, r.json().get("message") or r.text
            except Exception:
                return False, r.text
        return True, "sent"
    except Exception as e:
        return False, str(e)


# ==========================
# Trial config
# ==========================
def _trial_days() -> int:
    try:
        return max(1, int((os.getenv("TRIAL_DAYS", "14") or "14").strip()))
    except Exception:
        return 14


# ==========================
# Auth token (simple HMAC-signed JSON)
# ==========================
AUTH_SIGNING_SECRET = (os.getenv("AUTH_SIGNING_SECRET", "") or "").strip()


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def _sign(payload_bytes: bytes) -> str:
    key = (AUTH_SIGNING_SECRET or "dev-secret-change-me").encode("utf-8")
    return _b64url_encode(hmac.new(key, payload_bytes, hashlib.sha256).digest())


def _make_token(email: str, trial_ends_epoch: int) -> str:
    payload = {
        "email": email,
        "trial_ends": int(trial_ends_epoch),
        "iat": int(_now()),
        "v": 1,
    }
    pb = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"{_b64url_encode(pb)}.{_sign(pb)}"


def _verify_token(token: str) -> Optional[dict]:
    try:
        if not token or "." not in token:
            return None
        p, sig = token.split(".", 1)
        pb = _b64url_decode(p)
        expected = _sign(pb)
        if not hmac.compare_digest(expected, sig):
            return None
        payload = json.loads(pb.decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        if "email" not in payload or "trial_ends" not in payload:
            return None
        return payload
    except Exception:
        return None


# ==========================
# OTP helpers (Supabase stored, hashed)
# ==========================
def _email_ok(email: str) -> bool:
    if not email:
        return False
    email = email.strip()
    if len(email) > 254:
        return False
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email))


def _otp_code() -> str:
    # 6-digit numeric code
    n = int.from_bytes(os.urandom(4), "big") % 1000000
    return f"{n:06d}"


def _otp_hash(email: str, code: str) -> str:
    # stable per email+code (prevents rainbow reuse)
    key = (AUTH_SIGNING_SECRET or "dev-secret-change-me").encode("utf-8")
    msg = (email.strip().lower() + ":" + code.strip()).encode("utf-8")
    return hashlib.sha256(hmac.new(key, msg, hashlib.sha256).digest()).hexdigest()


def _trial_row(email: str) -> Optional[dict]:
    return _sb_select_one("fxco_trials", {"email": email})


def _trial_active(email: str) -> Tuple[bool, int]:
    """
    Returns (active, trial_ends_epoch)
    """
    row = _trial_row(email)
    if not row:
        return False, 0
    te = row.get("trial_ends")
    if not te:
        return False, 0
    dt = _parse_iso(str(te))
    if not dt:
        return False, 0
    epoch = int(dt.timestamp())
    return (_now() < epoch), epoch


# ==========================
# Paystack config + state (Supabase persistent)
# ==========================
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY", "").strip()
PAYSTACK_PUBLIC_KEY = os.getenv("PAYSTACK_PUBLIC_KEY", "").strip()
PAYSTACK_CURRENCY = (os.getenv("PAYSTACK_CURRENCY", "NGN") or "NGN").strip().upper()
PAYSTACK_BASE = "https://api.paystack.co"


def _require_payment():
    # support BOTH env names
    v = (os.getenv("PAYSTACK_REQUIRE_PAYMENT", "") or os.getenv("REQUIRE_PAYMENT", "1") or "1").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _access_days():
    try:
        return max(1, int((os.getenv("PAYSTACK_ACCESS_DAYS", "30") or "30").strip()))
    except Exception:
        return 30


def _paystack_amount_minor():
    """
    Supports:
      - PAYSTACK_AMOUNT=1000000 (minor units)
      - PAYSTACK_AMOUNT_NGN=10000 (major NGN)
    """
    amt_minor = (os.getenv("PAYSTACK_AMOUNT", "") or "").strip()
    if amt_minor:
        try:
            return max(0, int(amt_minor))
        except Exception:
            pass

    # fallback to NGN major
    try:
        major = int((os.getenv("PAYSTACK_AMOUNT_NGN", "10000") or "10000").strip())
    except Exception:
        major = 10000
    return max(0, major) * 100


def _paystack_headers():
    return {
        "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


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


def _get_paid_until(client_id: str) -> int:
    if not client_id:
        return 0
    row = _sb_select_one("fxco_access", {"client_id": client_id})
    if not row:
        return 0
    pu = row.get("paid_until")
    if not pu:
        return 0
    dt = _parse_iso(str(pu))
    if not dt:
        return 0
    return int(dt.timestamp())


def _is_client_unlocked(client_id: str) -> bool:
    if not _require_payment():
        return True
    if not client_id:
        return False
    return _now() < _get_paid_until(client_id)


def _set_paid_access(client_id: str, paid_until_epoch: int, ref: str):
    if not client_id:
        return
    dt = datetime.fromtimestamp(int(paid_until_epoch), tz=timezone.utc)
    _sb_upsert(
        "fxco_access",
        {
            "client_id": client_id,
            "paid_until": _dt_to_iso(dt),
            "last_ref": ref or None,
            "updated_at": _dt_to_iso(_utc_now_dt()),
        },
        conflict="client_id",
    )


# ==========================
# Rate limiting (in-memory, but keyed by client id when possible)
# ==========================
def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int((os.getenv(name, str(default)) or str(default)).strip()))
    except Exception:
        return default


FREE_LIMIT_60S = _env_int("FREE_LIMIT_60S", 5)
FREE_LIMIT_24H = _env_int("FREE_LIMIT_24H", 50)
PAID_LIMIT_60S = _env_int("PAID_LIMIT_60S", 60)
PAID_LIMIT_24H = _env_int("PAID_LIMIT_24H", 2000)

_SHORT_WINDOW_SECONDS = 60
_DAY_WINDOW_SECONDS = 24 * 60 * 60

_RATE_SHORT = {}
_RATE_DAY = {}


def _rate_check(identity_key: str, is_paid: bool):
    """
    Returns: (ok: bool, retry_after_seconds: int, headers: dict)
    Limits depend on paid status.
    """
    now = _now()

    short_limit = PAID_LIMIT_60S if is_paid else FREE_LIMIT_60S
    day_limit = PAID_LIMIT_24H if is_paid else FREE_LIMIT_24H

    if identity_key not in _RATE_SHORT:
        _RATE_SHORT[identity_key] = deque()
    if identity_key not in _RATE_DAY:
        _RATE_DAY[identity_key] = deque()

    short_q = _RATE_SHORT[identity_key]
    day_q = _RATE_DAY[identity_key]

    while short_q and (now - short_q[0]) >= _SHORT_WINDOW_SECONDS:
        short_q.popleft()
    while day_q and (now - day_q[0]) >= _DAY_WINDOW_SECONDS:
        day_q.popleft()

    short_remaining = max(0, short_limit - len(short_q))
    day_remaining = max(0, day_limit - len(day_q))

    if short_remaining <= 0 or day_remaining <= 0:
        retry_after = 1
        if short_remaining <= 0 and short_q:
            retry_after = max(retry_after, int(_SHORT_WINDOW_SECONDS - (now - short_q[0])) + 1)
        if day_remaining <= 0 and day_q:
            retry_after = max(retry_after, int(_DAY_WINDOW_SECONDS - (now - day_q[0])) + 1)

        headers = {
            "Retry-After": str(retry_after),
            "X-RateLimit-Plan": "paid" if is_paid else "free",
            "X-RateLimit-Limit-60s": str(short_limit),
            "X-RateLimit-Remaining-60s": str(short_remaining),
            "X-RateLimit-Limit-24h": str(day_limit),
            "X-RateLimit-Remaining-24h": str(day_remaining),
        }
        return False, retry_after, headers

    short_q.append(now)
    day_q.append(now)

    headers = {
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
    return jsonify(
        {
            "ok": True,
            "supabase_ok": _sb_ok(),
            "resend_ok": _resend_ok(),
            "require_payment": _require_payment(),
        }
    ), 200


# ==========================
# Auth (trial) endpoints
# ==========================
@app.post("/api/auth/start")
def auth_start():
    """
    POST { email }
    - sends OTP via Resend
    - stores OTP hash + expiry in Supabase
    - anti-spam: only 1 OTP per email per 60s
    """
    if not _sb_ok():
        return jsonify({"ok": False, "error": "Supabase not configured on server."}), 500
    if not _resend_ok():
        return jsonify({"ok": False, "error": "Resend not configured on server."}), 500
    if not AUTH_SIGNING_SECRET:
        return jsonify({"ok": False, "error": "Missing AUTH_SIGNING_SECRET on server."}), 500

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not _email_ok(email):
        return jsonify({"ok": False, "error": "Invalid email."}), 400

    # If trial already exists and still active -> don't spam OTP, just say ok
    active, trial_ends = _trial_active(email)
    if active:
        return jsonify({"ok": True, "already_active": True, "trial_ends": trial_ends}), 200

    # OTP resend cooldown
    otp_row = _sb_select_one("fxco_otps", {"email": email})
    if otp_row:
        sent_at = _parse_iso(str(otp_row.get("sent_at") or ""))
        if sent_at:
            if (_utc_now_dt() - sent_at) < timedelta(seconds=60):
                return jsonify({"ok": True, "cooldown": 60}), 200

    code = _otp_code()
    code_hash = _otp_hash(email, code)
    expires = _utc_now_dt() + timedelta(minutes=10)

    # upsert OTP row
    _sb_upsert(
        "fxco_otps",
        {
            "email": email,
            "code_hash": code_hash,
            "expires_at": _dt_to_iso(expires),
            "sent_at": _dt_to_iso(_utc_now_dt()),
            "attempts": 0,
        },
        conflict="email",
    )

    subject = "Your FXCO-PILOT verification code"
    html = f"""
      <div style="font-family:Arial,sans-serif;line-height:1.5">
        <h2>FXCO-PILOT • Email Verification</h2>
        <p>Your 6-digit code is:</p>
        <div style="font-size:28px;font-weight:800;letter-spacing:4px;padding:12px 16px;background:#111827;color:#fff;display:inline-block;border-radius:12px">
          {code}
        </div>
        <p style="margin-top:14px">This code expires in <b>10 minutes</b>.</p>
        <p style="opacity:.7">If you didn’t request this, you can ignore this email.</p>
      </div>
    """

    ok, msg = _send_email_resend(email, subject, html)
    if not ok:
        return jsonify({"ok": False, "error": f"Failed to send email: {msg}"}), 502

    return jsonify({"ok": True}), 200


@app.post("/api/auth/verify")
def auth_verify():
    """
    POST { email, code }
    - verifies OTP
    - creates trial if not existing
    - returns { token, trial_ends }
    """
    if not _sb_ok():
        return jsonify({"ok": False, "error": "Supabase not configured on server."}), 500
    if not AUTH_SIGNING_SECRET:
        return jsonify({"ok": False, "error": "Missing AUTH_SIGNING_SECRET on server."}), 500

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    code = (data.get("code") or "").strip()
    if not _email_ok(email):
        return jsonify({"ok": False, "error": "Invalid email."}), 400
    if not re.fullmatch(r"\d{6}", code or ""):
        return jsonify({"ok": False, "error": "Invalid code format."}), 400

    otp_row = _sb_select_one("fxco_otps", {"email": email})
    if not otp_row:
        return jsonify({"ok": False, "error": "No OTP found. Please request a new code."}), 400

    expires_at = _parse_iso(str(otp_row.get("expires_at") or ""))
    if not expires_at or _utc_now_dt() > expires_at:
        return jsonify({"ok": False, "error": "Code expired. Please request a new code."}), 400

    attempts = int(otp_row.get("attempts") or 0)
    if attempts >= 8:
        return jsonify({"ok": False, "error": "Too many attempts. Please request a new code."}), 429

    expected_hash = str(otp_row.get("code_hash") or "")
    got_hash = _otp_hash(email, code)

    if not expected_hash or not hmac.compare_digest(expected_hash, got_hash):
        _sb_update("fxco_otps", {"email": email}, {"attempts": attempts + 1})
        return jsonify({"ok": False, "error": "Incorrect code."}), 400

    # If trial already exists, keep existing end (don’t reset abuse)
    trial_row = _trial_row(email)
    if trial_row:
        te = _parse_iso(str(trial_row.get("trial_ends") or ""))
        if te:
            trial_ends_epoch = int(te.timestamp())
        else:
            trial_ends_epoch = 0
    else:
        te = _utc_now_dt() + timedelta(days=_trial_days())
        trial_ends_epoch = int(te.timestamp())
        cid = _get_client_id_header() or None
        ip = _client_ip()

        _sb_upsert(
            "fxco_trials",
            {
                "email": email,
                "trial_ends": _dt_to_iso(te),
                "created_at": _dt_to_iso(_utc_now_dt()),
                "last_client_id": cid,
                "last_ip": ip,
            },
            conflict="email",
        )

    # OTP consumed (optional: delete; we just expire it immediately)
    _sb_update("fxco_otps", {"email": email}, {"expires_at": _dt_to_iso(_utc_now_dt())})

    token = _make_token(email, trial_ends_epoch)
    return jsonify({"ok": True, "token": token, "trial_ends": trial_ends_epoch}), 200


# ==========================
# Paystack endpoints
# ==========================
@app.get("/api/paystack/config")
def paystack_config():
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

    callback_url = _ensure_paystack_success_param(_callback_url())

    payload = {
        "email": email,
        "amount": int(_paystack_amount_minor()),
        "currency": PAYSTACK_CURRENCY,
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

        data = j.get("data") or {}
        return jsonify({"ok": True, "authorization_url": data.get("authorization_url"), "reference": data.get("reference")}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/paystack/verify")
def paystack_verify():
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

        paid = (status == "success") and (currency == PAYSTACK_CURRENCY) and (amount >= int(_paystack_amount_minor()))

        meta = data.get("metadata") or {}
        client_id = (meta.get("fxco_client_id") or "").strip()
        if not client_id:
            client_id = _get_client_id_header() or ("ip:" + _client_ip())

        if paid and client_id:
            paid_until = int(_now() + _access_days() * 24 * 60 * 60)
            _set_paid_access(client_id, paid_until, reference)
            return jsonify({"ok": True, "paid": True, "reference": reference, "paid_until": paid_until}), 200

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
        r = requests.get(f"{TD_BASE}/price", params={"symbol": symbol, "apikey": TWELVE_DATA_API_KEY}, timeout=10)
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
            params={"symbol": symbol, "interval": interval, "outputsize": limit, "apikey": TWELVE_DATA_API_KEY},
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

    vals = list(reversed(values))
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
    client_id = _get_client_id_header() or ("ip:" + ip)

    # Determine trial status from token
    token = (request.headers.get("X-FXCO-Auth") or "").strip()
    token_payload = _verify_token(token) if token else None

    trial_active = False
    trial_ends_epoch = 0
    if token_payload and isinstance(token_payload, dict):
        email = str(token_payload.get("email") or "").strip().lower()
        trial_ends_epoch = int(token_payload.get("trial_ends") or 0)
        # Always confirm server-side (Supabase) so localStorage can't fake it
        active, te = _trial_active(email)
        trial_active = active
        trial_ends_epoch = te or trial_ends_epoch

    paid_active = _is_client_unlocked(client_id)

    # Gate: if require_payment, allow only (paid OR trial)
    if _require_payment() and not paid_active and not trial_active:
        return jsonify({"error": "Access locked. Start free trial (verify email) or unlock via Pricing."}), 402

    # Identity for rate limiting (not IP-only)
    identity_key = f"cid:{client_id}"
    # Apply PAID limits only if paid; trial uses FREE limits
    ok, retry_after, rl_headers = _rate_check(identity_key, is_paid=paid_active)
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
        if trial_ends_epoch:
            resp.headers["X-FXCO-Trial-Ends"] = str(trial_ends_epoch)
        return resp

    # OPTIONAL: test mode (cheap)
    if request.args.get("test") == "1":
        resp = jsonify({"ok": True, "mode": "rate_limit_test"})
        for k, v in rl_headers.items():
            resp.headers[k] = v
        if trial_ends_epoch:
            resp.headers["X-FXCO-Trial-Ends"] = str(trial_ends_epoch)
        return resp, 200

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
        if trial_ends_epoch:
            resp.headers["X-FXCO-Trial-Ends"] = str(trial_ends_epoch)
        return resp

    oa = _get_openai_client()
    if oa is None:
        resp = jsonify({"error": "Missing OPENAI_API_KEY on server."})
        resp.status_code = 500
        for k, v in rl_headers.items():
            resp.headers[k] = v
        if trial_ends_epoch:
            resp.headers["X-FXCO-Trial-Ends"] = str(trial_ends_epoch)
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
        if trial_ends_epoch:
            resp.headers["X-FXCO-Trial-Ends"] = str(trial_ends_epoch)
        return resp

    except json.JSONDecodeError:
        resp = jsonify({"error": "Model did not return valid JSON."})
        resp.status_code = 502
        for k, v in rl_headers.items():
            resp.headers[k] = v
        if trial_ends_epoch:
            resp.headers["X-FXCO-Trial-Ends"] = str(trial_ends_epoch)
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
        if trial_ends_epoch:
            resp.headers["X-FXCO-Trial-Ends"] = str(trial_ends_epoch)
        return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
