import base64
import hashlib
import hmac
import json
import os
import re
import time
import uuid
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Any, Dict, List

import requests
from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)

# ============================================================
# Logging (Render)
# ============================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fxco-pilot")

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
# Supabase (for OTP + trials + paid access persistence + report shares):
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
#
# Shareable reports:
#   REPORT_SHARE_TTL_DAYS=30
#
# Upstash Redis REST:
#   UPSTASH_REDIS_REST_URL=https://xxxx.upstash.io
#   UPSTASH_REDIS_REST_TOKEN=xxxx
# ============================================================

# ==========================
# Request lifecycle helpers
# ==========================
def _make_request_id() -> str:
    return str(uuid.uuid4())


@app.before_request
def _before_request():
    request._fxco_start_ms = int(time.time() * 1000)
    request._fxco_req_id = request.headers.get("X-Request-Id", "").strip() or _make_request_id()


def _cors_headers():
    allowed = (os.getenv("ALLOWED_ORIGINS", "") or "").strip()
    origin = (request.headers.get("Origin") or "").strip()

    if allowed:
        allowed_list = [x.strip() for x in allowed.split(",") if x.strip()]
        allow_origin = origin if origin in allowed_list else (allowed_list[0] if allowed_list else "*")
    else:
        allow_origin = origin or "*"

    return {
        "Access-Control-Allow-Origin": allow_origin,
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type,Authorization,X-FXCO-Auth,X-FXCO-Client,X-Request-Id",
        "Access-Control-Expose-Headers": "X-Request-Id,X-RateLimit-Plan,X-RateLimit-Limit-60s,X-RateLimit-Remaining-60s,X-RateLimit-Limit-24h,X-RateLimit-Remaining-24h,X-FXCO-Trial-Ends",
        "Access-Control-Max-Age": "86400",
        "Vary": "Origin",
    }


@app.after_request
def _after_request(resp):
    try:
        req_id = getattr(request, "_fxco_req_id", None) or _make_request_id()
        resp.headers["X-Request-Id"] = req_id

        if request.path.startswith("/api/") or request.path == "/health":
            for k, v in _cors_headers().items():
                resp.headers.setdefault(k, v)

        start_ms = getattr(request, "_fxco_start_ms", None)
        if start_ms:
            dur = int(time.time() * 1000) - int(start_ms)
            resp.headers.setdefault("Server-Timing", f"app;dur={dur}")

        try:
            dur_ms = int(time.time() * 1000) - int(start_ms or int(time.time() * 1000))
            logger.info("%s %s %s %sms rid=%s", request.method, request.path, resp.status_code, dur_ms, req_id)
        except Exception:
            pass

    except Exception:
        pass

    return resp


@app.route("/api/<path:_path>", methods=["OPTIONS"])
def _api_options(_path):
    resp = jsonify({"ok": True})
    return resp, 200


@app.errorhandler(Exception)
def _handle_exception(e):
    req_id = getattr(request, "_fxco_req_id", None) or _make_request_id()
    logger.exception("Unhandled exception rid=%s path=%s", req_id, request.path)
    resp = jsonify({"ok": False, "error": "Internal server error", "request_id": req_id})
    return resp, 500


# ==========================
# Time helpers
# ==========================
def _now() -> float:
    return time.time()


def _utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _dt_to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _utc_now_iso_z() -> str:
    return _utc_now_dt().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fx_session_from_utc(dt_utc: datetime) -> str:
    h = dt_utc.hour
    if 0 <= h < 8:
        return "Asia"
    if 8 <= h < 13:
        return "London"
    if 13 <= h < 22:
        return "New York"
    return "Off-hours"


def _parse_iso(dt_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _ms_now() -> int:
    return int(_now() * 1000)


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
# Upstash Redis (REST) — institutional cache + rate limit backing
# ==========================
UPSTASH_REDIS_REST_URL = (os.getenv("UPSTASH_REDIS_REST_URL", "") or "").strip().rstrip("/")
UPSTASH_REDIS_REST_TOKEN = (os.getenv("UPSTASH_REDIS_REST_TOKEN", "") or "").strip()


def _redis_ok() -> bool:
    return bool(UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN)


def _redis_headers():
    return {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}", "Content-Type": "application/json"}


def _redis_cmd(command: List[Any], timeout: int = 8) -> Optional[Any]:
    """
    Upstash REST supports POST {url} with JSON: {"command": ["PING"]} etc.
    Returns 'result' on success, else None.
    """
    if not _redis_ok():
        return None
    try:
        r = requests.post(
            UPSTASH_REDIS_REST_URL,
            headers=_redis_headers(),
            data=json.dumps({"command": command}),
            timeout=timeout,
        )
        if r.status_code >= 400:
            return None
        j = r.json()
        return j.get("result")
    except Exception:
        return None


def _redis_ping() -> bool:
    res = _redis_cmd(["PING"])
    return (str(res or "")).upper() == "PONG"


def _redis_get(key: str) -> Optional[str]:
    res = _redis_cmd(["GET", key])
    if res is None:
        return None
    # Upstash returns string or null
    return res


def _redis_setex(key: str, ttl_seconds: int, value: str) -> bool:
    res = _redis_cmd(["SETEX", key, int(ttl_seconds), value])
    return (str(res or "")).upper() == "OK"


def _redis_incr_with_ttl(key: str, ttl_seconds: int) -> Optional[int]:
    """
    Fixed-window counter:
    - INCR key
    - On first increment, set EXPIRE
    """
    v = _redis_cmd(["INCR", key])
    if v is None:
        return None
    try:
        iv = int(v)
    except Exception:
        return None
    if iv == 1:
        _redis_cmd(["EXPIRE", key, int(ttl_seconds)])
    return iv


# ==========================
# Simple cache (Redis-first, fallback memory)
# ==========================
_CACHE: Dict[str, Any] = {}
_CACHE_TTL: Dict[str, float] = {}


def _cache_get(key: str):
    # Redis first
    if _redis_ok():
        raw = _redis_get(f"fxco:cache:{key}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return raw

    # Memory fallback
    if key in _CACHE and time.time() < _CACHE_TTL.get(key, 0):
        return _CACHE[key]
    return None


def _cache_set(key: str, value, ttl: int = 60):
    # Redis first
    if _redis_ok():
        try:
            payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            payload = json.dumps({"value": str(value)}, separators=(",", ":"), ensure_ascii=False)
        _redis_setex(f"fxco:cache:{key}", int(ttl), payload)
        return

    # Memory fallback
    _CACHE[key] = value
    _CACHE_TTL[key] = time.time() + ttl


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
    if not _sb_ok():
        return None
    params = {k: f"eq.{v}" for k, v in filters.items()}
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


def _sb_insert_returning(table: str, row: dict) -> Optional[dict]:
    if not _sb_ok():
        return None
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={**_sb_headers(), "Prefer": "return=representation"},
            data=json.dumps(row),
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
            data=json.dumps({"from": RESEND_FROM_EMAIL, "to": [to_email], "subject": subject, "html": html}),
            timeout=12,
        )
        if r.status_code >= 400:
            try:
                return False, (r.json() or {}).get("message") or r.text
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
# Shareable report config
# ==========================
def _report_share_ttl_days() -> int:
    try:
        return max(1, int((os.getenv("REPORT_SHARE_TTL_DAYS", "30") or "30").strip()))
    except Exception:
        return 30


def _origin_base_url() -> str:
    xf_host = (request.headers.get("X-Forwarded-Host") or "").strip()
    xf_proto = (request.headers.get("X-Forwarded-Proto") or "").strip()
    if xf_host:
        proto = xf_proto or "https"
        return f"{proto}://{xf_host}".rstrip("/")

    origin = (request.headers.get("Origin") or "").strip()
    if origin:
        return origin.rstrip("/")

    ref = (request.headers.get("Referer") or "").strip()
    if ref:
        try:
            u = ref.split("#", 1)[0]
            u = u.split("?", 1)[0]
            parts = u.split("/")
            if len(parts) >= 3:
                return (parts[0] + "//" + parts[2]).rstrip("/")
        except Exception:
            pass

    return request.host_url.rstrip("/")


def _is_uuid_like(s: str) -> bool:
    try:
        uuid.UUID(str(s))
        return True
    except Exception:
        return False


def _store_share_report(report_obj: dict, client_id: str, email: Optional[str]) -> Optional[dict]:
    if not _sb_ok():
        return None

    rid = str(uuid.uuid4())
    expires_at = _utc_now_dt() + timedelta(days=_report_share_ttl_days())

    row = {
        "id": rid,
        "report": report_obj,
        "client_id": client_id or None,
        "email": (email or "").strip().lower() or None,
        "created_at": _dt_to_iso(_utc_now_dt()),
        "expires_at": _dt_to_iso(expires_at),
    }
    created = _sb_insert_returning("fxco_reports", row)
    if not created:
        row2 = dict(row)
        row2.pop("id", None)
        created = _sb_insert_returning("fxco_reports", row2)

    if not created:
        return None

    out_id = str(created.get("id") or rid)
    out_expires = str(created.get("expires_at") or _dt_to_iso(expires_at))
    return {"id": out_id, "expires_at": out_expires}


def _load_share_report(share_id: str) -> Optional[dict]:
    if not _sb_ok():
        return None
    row = _sb_select_one("fxco_reports", {"id": share_id})
    if not row:
        return None

    exp = _parse_iso(str(row.get("expires_at") or ""))
    if exp and _utc_now_dt() > exp:
        return None

    report = row.get("report")
    if not isinstance(report, dict):
        return None
    return report


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
    payload = {"email": email, "trial_ends": int(trial_ends_epoch), "iat": int(_now()), "v": 1}
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
    n = int.from_bytes(os.urandom(4), "big") % 1000000
    return f"{n:06d}"


def _otp_hash(email: str, code: str) -> str:
    key = (AUTH_SIGNING_SECRET or "dev-secret-change-me").encode("utf-8")
    msg = (email.strip().lower() + ":" + code.strip()).encode("utf-8")
    return hashlib.sha256(hmac.new(key, msg, hashlib.sha256).digest()).hexdigest()


def _trial_row(email: str) -> Optional[dict]:
    return _sb_select_one("fxco_trials", {"email": email})


def _trial_active(email: str) -> Tuple[bool, int]:
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
    v = (os.getenv("PAYSTACK_REQUIRE_PAYMENT", "") or os.getenv("REQUIRE_PAYMENT", "1") or "1").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _access_days():
    try:
        return max(1, int((os.getenv("PAYSTACK_ACCESS_DAYS", "30") or "30").strip()))
    except Exception:
        return 30


def _paystack_amount_minor():
    amt_minor = (os.getenv("PAYSTACK_AMOUNT", "") or "").strip()
    if amt_minor:
        try:
            return max(0, int(amt_minor))
        except Exception:
            pass

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
        {"client_id": client_id, "paid_until": _dt_to_iso(dt), "last_ref": ref or None, "updated_at": _dt_to_iso(_utc_now_dt())},
        conflict="client_id",
    )


# ==========================
# Rate limiting (Redis-first, fixed window; fallback memory)
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

_RATE_SHORT: Dict[str, deque] = {}
_RATE_DAY: Dict[str, deque] = {}


def _rate_check(identity_key: str, is_paid: bool):
    now = _now()

    short_limit = PAID_LIMIT_60S if is_paid else FREE_LIMIT_60S
    day_limit = PAID_LIMIT_24H if is_paid else FREE_LIMIT_24H

    plan = "paid" if is_paid else "free"

    # Redis implementation (recommended)
    if _redis_ok():
        # Fixed windows: bucket by current minute/day
        minute_bucket = int(now // _SHORT_WINDOW_SECONDS)
        day_bucket = int(now // _DAY_WINDOW_SECONDS)

        k60 = f"fxco:rl:{plan}:60s:{identity_key}:{minute_bucket}"
        k24 = f"fxco:rl:{plan}:24h:{identity_key}:{day_bucket}"

        c60 = _redis_incr_with_ttl(k60, _SHORT_WINDOW_SECONDS + 2)
        c24 = _redis_incr_with_ttl(k24, _DAY_WINDOW_SECONDS + 60)

        # If Redis hiccups, fall back to memory
        if c60 is None or c24 is None:
            logger.warning("redis_rl_failed identity=%s", identity_key)
        else:
            remaining_60 = max(0, short_limit - c60)
            remaining_24 = max(0, day_limit - c24)

            if c60 > short_limit or c24 > day_limit:
                retry_after = 1
                # In fixed-window, retry is until window rollover
                retry_after = max(retry_after, int((_SHORT_WINDOW_SECONDS - (now % _SHORT_WINDOW_SECONDS))) + 1) if c60 > short_limit else retry_after
                retry_after = max(retry_after, int((_DAY_WINDOW_SECONDS - (now % _DAY_WINDOW_SECONDS))) + 1) if c24 > day_limit else retry_after

                headers = {
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Plan": plan,
                    "X-RateLimit-Limit-60s": str(short_limit),
                    "X-RateLimit-Remaining-60s": str(max(0, short_limit - min(c60, short_limit))),
                    "X-RateLimit-Limit-24h": str(day_limit),
                    "X-RateLimit-Remaining-24h": str(max(0, day_limit - min(c24, day_limit))),
                }
                return False, retry_after, headers

            headers = {
                "X-RateLimit-Plan": plan,
                "X-RateLimit-Limit-60s": str(short_limit),
                "X-RateLimit-Remaining-60s": str(remaining_60),
                "X-RateLimit-Limit-24h": str(day_limit),
                "X-RateLimit-Remaining-24h": str(remaining_24),
            }
            return True, 0, headers

    # Memory fallback (single instance)
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
            "X-RateLimit-Plan": plan,
            "X-RateLimit-Limit-60s": str(short_limit),
            "X-RateLimit-Remaining-60s": str(short_remaining),
            "X-RateLimit-Limit-24h": str(day_limit),
            "X-RateLimit-Remaining-24h": str(day_remaining),
        }
        return False, retry_after, headers

    short_q.append(now)
    day_q.append(now)

    headers = {
        "X-RateLimit-Plan": plan,
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
    # Real ping test (so redis_ok reflects reality)
    redis_ok = False
    try:
        redis_ok = _redis_ping() if _redis_ok() else False
    except Exception:
        redis_ok = False

    openai_ok = bool(os.getenv("OPENAI_API_KEY", "").strip())
    twelvedata_ok = bool(TWELVE_DATA_API_KEY)

    return jsonify(
        {
            "ok": True,
            "openai_ok": openai_ok,
            "twelvedata_ok": twelvedata_ok,
            "redis_ok": redis_ok,
            "supabase_ok": _sb_ok(),
            "resend_ok": _resend_ok(),
            "require_payment": _require_payment(),
        }
    ), 200


# ==========================
# Shareable report endpoints
# ==========================
@app.post("/api/report/share")
def report_share():
    if not _sb_ok():
        return jsonify({"ok": False, "error": "Supabase not configured on server."}), 500

    data = request.get_json(silent=True) or {}
    report_obj = data.get("report") if isinstance(data, dict) else None
    if not isinstance(report_obj, dict):
        if isinstance(data, dict):
            report_obj = data

    if not isinstance(report_obj, dict) or not report_obj:
        return jsonify({"ok": False, "error": "Missing report payload."}), 400

    client_id = _get_client_id_header() or ("ip:" + _client_ip())
    token = (request.headers.get("X-FXCO-Auth") or "").strip()
    token_payload = _verify_token(token) if token else None
    email = None
    if token_payload and isinstance(token_payload, dict):
        email = str(token_payload.get("email") or "").strip().lower() or None

    created = _store_share_report(report_obj, client_id=client_id, email=email)
    if not created:
        return jsonify({"ok": False, "error": "Failed to create share report (db insert failed)."}), 502

    share_id = created["id"]
    share_url = f"{_origin_base_url()}/result.html?r={share_id}"
    return jsonify({"ok": True, "share_id": share_id, "share_url": share_url}), 200


@app.get("/api/report/<share_id>")
def report_get(share_id: str):
    if not _sb_ok():
        return jsonify({"ok": False, "error": "Supabase not configured on server."}), 500

    sid = (share_id or "").strip()
    if not sid or not _is_uuid_like(sid):
        return jsonify({"ok": False, "error": "Invalid report id."}), 400

    report_obj = _load_share_report(sid)
    if not report_obj:
        return jsonify({"ok": False, "error": "Report not found (or expired)."}), 404

    if isinstance(report_obj, dict):
        report_obj.setdefault("share_id", sid)
        report_obj.setdefault("share_url", f"{_origin_base_url()}/result.html?r={sid}")
        try:
            if isinstance(report_obj.get("analysis"), dict):
                report_obj["analysis"].setdefault("share_id", sid)
                report_obj["analysis"].setdefault("share_url", f"{_origin_base_url()}/result.html?r={sid}")
        except Exception:
            pass

    return jsonify({"ok": True, "report": report_obj}), 200


# ==========================
# Auth (trial) endpoints
# ==========================
@app.post("/api/auth/start")
def auth_start():
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

    active, trial_ends = _trial_active(email)
    if active:
        return jsonify({"ok": True, "already_active": True, "trial_ends": trial_ends}), 200

    otp_row = _sb_select_one("fxco_otps", {"email": email})
    if otp_row:
        sent_at = _parse_iso(str(otp_row.get("sent_at") or ""))
        if sent_at and (_utc_now_dt() - sent_at) < timedelta(seconds=60):
            return jsonify({"ok": True, "cooldown": 60}), 200

    code = _otp_code()
    code_hash = _otp_hash(email, code)
    expires = _utc_now_dt() + timedelta(minutes=10)

    _sb_upsert(
        "fxco_otps",
        {"email": email, "code_hash": code_hash, "expires_at": _dt_to_iso(expires), "sent_at": _dt_to_iso(_utc_now_dt()), "attempts": 0},
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

    trial_row = _trial_row(email)
    if trial_row:
        te = _parse_iso(str(trial_row.get("trial_ends") or ""))
        trial_ends_epoch = int(te.timestamp()) if te else 0
    else:
        te = _utc_now_dt() + timedelta(days=_trial_days())
        trial_ends_epoch = int(te.timestamp())
        cid = _get_client_id_header() or None
        ip = _client_ip()

        _sb_upsert(
            "fxco_trials",
            {"email": email, "trial_ends": _dt_to_iso(te), "created_at": _dt_to_iso(_utc_now_dt()), "last_client_id": cid, "last_ip": ip},
            conflict="email",
        )

    _sb_update("fxco_otps", {"email": email}, {"expires_at": _dt_to_iso(_utc_now_dt())})

    token = _make_token(email, trial_ends_epoch)
    return jsonify({"ok": True, "token": token, "trial_ends": trial_ends_epoch}), 200


# ==========================
# Paystack endpoints (hardened)
# ==========================
@app.get("/api/paystack/config")
def paystack_config():
    try:
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
    except Exception as e:
        return jsonify({"ok": False, "require_payment": False, "currency": "NGN", "amount": 10000 * 100, "public_key": None, "access_days": 30, "error": str(e)}), 200


def _safe_json(r: requests.Response) -> dict:
    try:
        return r.json()
    except Exception:
        return {"status": False, "message": f"Non-JSON response: HTTP {r.status_code}", "raw": (r.text or "")[:500]}


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

    client_id = _get_client_id_header() or ("ip:" + _client_ip())
    callback_url = _ensure_paystack_success_param(_callback_url())

    payload = {
        "email": email,
        "amount": int(_paystack_amount_minor()),
        "currency": PAYSTACK_CURRENCY,
        "callback_url": callback_url,
        "metadata": {"fxco_client_id": client_id, "product": "FXCO-PILOT", "access_days": _access_days()},
    }

    try:
        r = requests.post(f"{PAYSTACK_BASE}/transaction/initialize", headers=_paystack_headers(), data=json.dumps(payload), timeout=15)
        j = _safe_json(r)
        if not j.get("status"):
            return jsonify({"ok": False, "error": j.get("message") or "Paystack init failed."}), 502

        d = j.get("data") or {}
        return jsonify({"ok": True, "authorization_url": d.get("authorization_url"), "reference": d.get("reference")}), 200
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
        j = _safe_json(r)
        if not j.get("status"):
            return jsonify({"ok": False, "error": j.get("message") or "Paystack verify failed."}), 502

        data = j.get("data") or {}
        status = (data.get("status") or "").lower()
        currency = (data.get("currency") or "").upper()
        amount = int(data.get("amount") or 0)

        paid = (status == "success") and (currency == PAYSTACK_CURRENCY) and (amount >= int(_paystack_amount_minor()))

        meta = data.get("metadata") or {}
        client_id = (meta.get("fxco_client_id") or "").strip() or (_get_client_id_header() or ("ip:" + _client_ip()))

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


def td_symbol(s: str) -> str:
    raw = (s or "").upper().strip()
    if "/" in raw:
        return raw
    sym = _norm_symbol(raw)
    if re.fullmatch(r"[A-Z]{6}", sym):
        return f"{sym[:3]}/{sym[3:]}"
    return sym


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
# Timeframe + duration
# ==========================
def _parse_duration_minutes(text: str) -> Optional[int]:
    t = (text or "").lower()

    m_only = re.search(r"\b(\d{1,4})\s*(m|min|mins|minute|minutes)\b", t)
    h = re.search(r"\b(\d{1,3})\s*(h|hr|hrs|hour|hours)\b", t)

    hours = int(h.group(1)) if h else 0
    mins = int(m_only.group(1)) if m_only else 0

    total = hours * 60 + mins
    if total > 0:
        return total

    just_num_min = re.search(r"\b(\d{2,4})\s*(minutes)\b", t)
    if just_num_min:
        try:
            return int(just_num_min.group(1))
        except Exception:
            return None

    return None


def _normalize_mode(tf: str) -> str:
    t = (tf or "").strip().lower()
    if t in ("scalp", "scalping"):
        return "scalp"
    if t in ("intraday", "intra"):
        return "intraday"
    if t in ("swing", "swinger", "swingtrade", "swing trading"):
        return "swing"
    return t or "intraday"


def _map_horizon_to_tfs(mode: str, hold_minutes: Optional[int]) -> Tuple[str, str]:
    mode = _normalize_mode(mode)

    if hold_minutes is None:
        if mode == "scalp":
            hold_minutes = 20
        elif mode == "intraday":
            hold_minutes = 135
        elif mode == "swing":
            hold_minutes = 1440
        else:
            hold_minutes = 135

    if hold_minutes <= 30:
        return "5min", "1min"
    if hold_minutes <= 90:
        return "15min", "5min"
    if hold_minutes <= 240:
        return "1h", "15min"
    return "4h", "1h"


# ==========================
# chart_tf (NEW): honor chart timeframe as structure driver
# ==========================
_TD_INTERVALS = ["1min", "5min", "15min", "30min", "1h", "4h", "1day"]


def _normalize_chart_tf(chart_tf: str) -> str:
    s = (chart_tf or "").strip().lower()
    if not s:
        return ""

    s = s.replace(" ", "")
    s = s.replace("minutes", "min").replace("minute", "min")
    s = s.replace("hours", "h").replace("hour", "h")
    s = s.replace("days", "day").replace("d", "day") if re.fullmatch(r"\d+d", s) else s

    if s in ("1m", "1min"):
        return "1min"
    if s in ("5m", "5min"):
        return "5min"
    if s in ("15m", "15min"):
        return "15min"
    if s in ("30m", "30min"):
        return "30min"
    if s in ("1h", "60min"):
        return "1h"
    if s in ("4h", "240min"):
        return "4h"
    if s in ("1d", "1day", "day"):
        return "1day"

    return ""


def _pick_execution_tf(structure_tf: str) -> str:
    try:
        i = _TD_INTERVALS.index(structure_tf)
    except Exception:
        return "15min"
    if i <= 0:
        return "1min"
    return _TD_INTERVALS[max(0, i - 1)]


# ==========================
# Twelve Data calls (cached)
# ==========================
def td_price(symbol: str):
    if not TWELVE_DATA_API_KEY:
        return {"ok": False, "error": "Missing TWELVE_DATA_API_KEY (live data disabled)."}

    symbol = td_symbol(symbol or "EURUSD")
    cache_key = f"td_price::{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    snap_dt = _utc_now_dt()
    snap_ts = snap_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    snap_session = _fx_session_from_utc(snap_dt)

    try:
        r = requests.get(f"{TD_BASE}/price", params={"symbol": symbol, "apikey": TWELVE_DATA_API_KEY}, timeout=10)
        data = r.json()
        if data.get("status") == "error":
            out = {"ok": False, "symbol": symbol, "error": data.get("message", "Twelve Data error"), "source": "twelvedata", "timestamp_utc": snap_ts, "session": snap_session}
            _cache_set(cache_key, out, ttl=15)
            return out

        p = float(data["price"])
        out = {"ok": True, "symbol": symbol, "price": p, "source": "twelvedata", "timestamp_utc": snap_ts, "session": snap_session}
        _cache_set(cache_key, out, ttl=15)
        return out
    except Exception as e:
        out = {"ok": False, "symbol": symbol, "error": str(e), "source": "twelvedata", "timestamp_utc": snap_ts, "session": snap_session}
        _cache_set(cache_key, out, ttl=15)
        return out


def td_candles(symbol: str, interval: str = "5min", limit: int = 120):
    if not TWELVE_DATA_API_KEY:
        return {"ok": False, "error": "Missing TWELVE_DATA_API_KEY (live data disabled)."}

    symbol = td_symbol(symbol or "EURUSD")
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


# ==========================
# Structure (UPGRADED): pivot swings + BOS detection
# ==========================
def _pivot_points(highs: List[float], lows: List[float], left: int = 2, right: int = 2):
    ph = []
    pl = []
    n = len(highs)
    for i in range(left, n - right):
        h = highs[i]
        l = lows[i]
        if all(h > highs[i - k] for k in range(1, left + 1)) and all(h > highs[i + k] for k in range(1, right + 1)):
            ph.append(i)
        if all(l < lows[i - k] for k in range(1, left + 1)) and all(l < lows[i + k] for k in range(1, right + 1)):
            pl.append(i)
    return ph, pl


def structure_from_candles(values):
    if not values or len(values) < 60:
        return {"structure": "unclear", "broken": False, "details": "Not enough candle data."}

    vals = list(reversed(values))  # oldest -> newest
    try:
        closes = [float(v["close"]) for v in vals]
        highs = [float(v["high"]) for v in vals]
        lows = [float(v["low"]) for v in vals]
    except Exception:
        return {"structure": "unclear", "broken": False, "details": "Candle parse error."}

    last_close = closes[-1]

    ph, pl = _pivot_points(highs, lows, left=2, right=2)
    if len(ph) < 2 or len(pl) < 2:
        prev = closes[-25]
        trend = "bullish" if last_close > prev else "bearish" if last_close < prev else "unclear"
        recent_low = min(lows[-15:])
        prior_low = min(lows[-40:-15])
        recent_high = max(highs[-15:])
        prior_high = max(highs[-40:-15])
        if trend == "bullish" and recent_low > prior_low:
            return {"structure": "bullish (HL)", "broken": False, "details": "Higher low detected (fallback)."}
        if trend == "bearish" and recent_high < prior_high:
            return {"structure": "bearish (LH)", "broken": False, "details": "Lower high detected (fallback)."}
        return {"structure": "unclear", "broken": False, "details": "Insufficient pivots; fallback trend unclear."}

    last_ph_i, prev_ph_i = ph[-1], ph[-2]
    last_pl_i, prev_pl_i = pl[-1], pl[-2]

    last_ph = highs[last_ph_i]
    prev_ph = highs[prev_ph_i]
    last_pl = lows[last_pl_i]
    prev_pl = lows[prev_pl_i]

    hh = last_ph > prev_ph
    lh = last_ph < prev_ph
    hl = last_pl > prev_pl
    ll = last_pl < prev_pl

    if hh and hl:
        struct = "bullish (HH/HL)"
    elif lh and ll:
        struct = "bearish (LH/LL)"
    elif hh and ll:
        struct = "expanding / volatile"
    elif lh and hl:
        struct = "range / compression"
    else:
        struct = "unclear"

    broken = False
    bos = ""
    if last_close > last_ph:
        broken = True
        bos = "BOS up (close > last swing high)"
    if last_close < last_pl:
        broken = True
        bos = "BOS down (close < last swing low)"

    details = f"last_close={last_close:.5f}, last_PH={last_ph:.5f}, last_PL={last_pl:.5f}"
    if bos:
        details += f", {bos}"

    return {"structure": struct, "broken": broken, "details": details}


# ==========================
# Confidence / reasons / warnings
# ==========================
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
    if any(x in liquidity for x in ["sweep", "grab", "equal", "liquidity", "stop run", "raid"]):
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
    return reasons[:6]


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
        if sl is not None and isinstance(price, (int, float)):
            if "long" in bias and price <= sl:
                warnings.append("Live price has hit SL => invalidated.")
            if "short" in bias and price >= sl:
                warnings.append("Live price has hit SL => invalidated.")

    if ("long" in bias and "bearish" in struct.lower()) or ("short" in bias and "bullish" in struct.lower()):
        warnings.append("Structure is against your bias (structure conflict).")

    return warnings[:10]


# ==========================
# Execution Plan (NEW): actionable outputs
# ==========================
def build_execution_plan(analysis: dict) -> dict:
    sc = analysis.get("signal_check", {}) or {}
    bias = (analysis.get("bias") or "Unclear").strip().lower()

    entry = _to_float(sc.get("entry"))
    sl = _to_float(sc.get("stop_loss"))
    targets = _parse_targets(sc.get("targets"))

    plan = {
        "invalidation_level": None,
        "aggressive_entry": None,
        "conservative_entry": None,
        "position_size_hint": "",
        "if_then_checklist": [],
    }

    if entry is None or sl is None:
        plan["if_then_checklist"].append("If entry or stop loss is missing, do not execute. Add levels first.")
        return plan

    plan["invalidation_level"] = sl

    risk = abs(entry - sl)
    if risk <= 0:
        plan["if_then_checklist"].append("If entry equals stop loss, the trade is invalid. Fix your risk.")
        return plan

    plan["aggressive_entry"] = entry

    if "long" in bias:
        plan["conservative_entry"] = round(entry - 0.25 * risk, 5)
    elif "short" in bias:
        plan["conservative_entry"] = round(entry + 0.25 * risk, 5)
    else:
        plan["conservative_entry"] = entry

    rr = sc.get("rr")
    if isinstance(rr, (int, float)):
        if rr >= 2.0:
            plan["position_size_hint"] = "RR is healthy. Consider normal risk size (still respect daily loss limits)."
        elif rr >= 1.5:
            plan["position_size_hint"] = "RR is acceptable. Consider smaller size unless the trigger is very clean."
        else:
            plan["position_size_hint"] = "RR is weak. Prefer smaller size or skip unless you can improve entry/TP."

    plan["if_then_checklist"].extend(
        [
            "If price tags entry but you don’t get a clean trigger (reclaim/confirm), wait.",
            "If structure conflicts with your bias, reduce size or skip.",
            "If price hits invalidation (SL), exit — no exceptions.",
        ]
    )
    if targets:
        plan["if_then_checklist"].append("If TP1 hits, consider moving SL to breakeven (only if structure supports).")

    return plan


# ==========================
# JSON reliability (UPGRADE #1): extract + retry + repair
# ==========================
def _extract_first_json_object(text: str) -> Optional[dict]:
    if not text:
        return None

    s = text.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        return None
    return None


def _repair_json_with_model(oa: OpenAI, raw_text: str, schema_hint: str) -> Optional[dict]:
    try:
        repair_prompt = f"""
Fix the following content into VALID JSON ONLY (no markdown, no extra text).
It MUST match this schema exactly:

{schema_hint}

Content to fix:
{raw_text}
"""
        completion = oa.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a strict JSON repair tool. Output ONLY valid JSON. No markdown."},
                {"role": "user", "content": repair_prompt},
            ],
            temperature=0.0,
        )
        out = completion.choices[0].message.content or ""
        obj = _extract_first_json_object(out)
        return obj
    except Exception:
        return None


# ==========================
# Payload parsing (UPDATED): includes chart_tf
# ==========================
def _pick_first(data, keys):
    for k in keys:
        v = None
        try:
            v = data.get(k)
        except Exception:
            v = None
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def _get_payload_fields():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        pair_type = _pick_first(data, ["pair_type", "pairType"])
        timeframe = _pick_first(data, ["timeframe", "timeframe_mode", "timeframeMode"])
        chart_tf = _pick_first(data, ["chart_tf", "chartTF", "chart_timeframe", "chartTimeframe"])
        signal_text = _pick_first(data, ["signal_input", "signal", "signalText"])
        return pair_type, timeframe, chart_tf, signal_text

    form = request.form or {}
    pair_type = _pick_first(form, ["pair_type", "pairType"])
    timeframe = _pick_first(form, ["timeframe", "timeframe_mode", "timeframeMode"])
    chart_tf = _pick_first(form, ["chart_tf", "chartTF", "chart_timeframe", "chartTimeframe"])
    signal_text = _pick_first(form, ["signal_input", "signal", "signalText"])
    return pair_type, timeframe, chart_tf, signal_text


# ==========================
# Institutional decision system (existing)
# ==========================
def _as_text(v) -> str:
    return (str(v or "")).strip()


def _institutional_score(analysis: dict) -> int:
    mc = analysis.get("market_context", {}) or {}
    sc = analysis.get("signal_check", {}) or {}

    structure = _as_text(mc.get("structure")).lower()
    liquidity = _as_text(mc.get("liquidity")).lower()
    alignment = _as_text(mc.get("timeframe_alignment")).lower()
    momentum = _as_text(mc.get("momentum")).lower()

    rr = sc.get("rr")
    direction = _as_text(sc.get("direction")).lower()

    score = 0

    if any(x in structure for x in ["bullish", "bearish", "trend", "clean", "hh/hl", "lh/ll"]):
        score += 20
    elif any(x in structure for x in ["range", "sideways", "chop", "unclear", "compression"]):
        score += 10
    elif structure:
        score += 12

    if any(x in liquidity for x in ["sweep", "grab", "raid", "liquidity taken", "stop run"]):
        score += 20
    elif any(x in liquidity for x in ["good", "ok", "healthy"]):
        score += 16
    elif liquidity:
        score += 10

    if any(x in alignment for x in ["aligned", "strong", "yes"]):
        score += 16
    elif any(x in alignment for x in ["mixed", "conflict", "partial"]):
        score += 9
    elif alignment:
        score += 8

    if direction in ("long", "short"):
        score += 8
    if sc.get("targets"):
        score += 3
    if any(x in momentum for x in ["strong", "impulsive"]):
        score += 4
    elif momentum:
        score += 2

    if isinstance(rr, (int, float)):
        if rr >= 2.5:
            score += 15
        elif rr >= 2.0:
            score += 12
        elif rr >= 1.5:
            score += 9
        elif rr >= 1.2:
            score += 5
        else:
            score += 2
    if sc.get("entry") is not None:
        score += 2
    if sc.get("stop_loss") is not None:
        score += 2

    return max(0, min(100, int(round(score))))


def institutional_decision_engine(analysis: dict) -> dict:
    hard_blocks = []
    conditions = []
    rationale = []

    mc = analysis.get("market_context", {}) or {}
    sc = analysis.get("signal_check", {}) or {}

    bias = _as_text(analysis.get("bias")).lower()
    structure = _as_text(mc.get("structure")).lower()
    liquidity = _as_text(mc.get("liquidity")).lower()
    alignment = _as_text(mc.get("timeframe_alignment")).lower()

    rr = sc.get("rr")

    horizon = analysis.get("assessment_horizon") or {}
    structure_tf = _as_text(horizon.get("structure_tf")).lower() or "unknown"
    execution_tf = _as_text(horizon.get("execution_tf")).lower() or "unknown"

    if sc.get("entry") is None:
        hard_blocks.append("Missing entry level.")
    if sc.get("stop_loss") is None:
        hard_blocks.append("Missing stop loss.")
    if not sc.get("targets"):
        hard_blocks.append("Missing targets / take-profit levels.")
    if isinstance(rr, (int, float)) and rr < 1.5:
        hard_blocks.append(f"Risk:Reward too low (RR≈{rr}). Minimum is 1.5 for execution.")

    if mc.get("structure_broken") is True:
        conditions.append("Break of structure detected recently: require a clean reclaim/confirmation before entry.")

    if any(x in liquidity for x in ["thin", "poor", "unknown", "low liquidity"]):
        hard_blocks.append("Liquidity conditions are poor/unclear.")

    if ("long" in bias and "bearish" in structure) or ("short" in bias and "bullish" in structure):
        hard_blocks.append("Bias conflicts with structure (HTF/LTF conflict).")

    inv = analysis.get("invalidation_warnings") or []
    if isinstance(inv, list):
        critical = [w for w in inv if any(k in _as_text(w).lower() for k in ["hit sl", "invalidated", "structure conflict"])]
        if critical:
            hard_blocks.append("Critical invalidation risk detected (see invalidation warnings).")

    score = _institutional_score(analysis)
    confidence = analysis.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0
    confidence = int(max(0, min(100, confidence)))

    if not hard_blocks:
        if any(x in alignment for x in ["mixed", "conflict", "partial"]):
            conditions.append("Wait for clearer timeframe alignment before executing.")
        if isinstance(rr, (int, float)) and rr < 2.0:
            conditions.append("Improve RR (aim ≥ 2.0) by refining entry or targets.")
        if any(x in structure for x in ["range", "compression"]):
            conditions.append("Range/compression: wait for sweep + reclaim or breakout + retest.")
        if not conditions and confidence < 65:
            conditions.append("Confidence is moderate: execute only with a clean trigger (confirm candle / reclaim).")

        why = analysis.get("why_this_trade") or []
        if isinstance(why, list):
            for x in why[:6]:
                t = _as_text(x)
                if t:
                    rationale.append(t)

    if hard_blocks:
        decision_tier = "DO_NOT_TRADE"
        decision_label = "AVOID TRADE"
    else:
        if score >= 75 and confidence >= 65:
            decision_tier = "EXECUTE"
            decision_label = "TAKE TRADE"
        elif score >= 60 or confidence >= 55:
            decision_tier = "EXECUTE_IF"
            decision_label = "TAKE IF..."
        elif score >= 45:
            decision_tier = "WAIT"
            decision_label = "WAIT"
        else:
            decision_tier = "DO_NOT_TRADE"
            decision_label = "AVOID TRADE"

    verdict_text = _as_text(analysis.get("verdict"))
    guidance = analysis.get("guidance") or []
    guidance_lines = []
    if isinstance(guidance, list):
        for g in guidance[:6]:
            gg = _as_text(g)
            if gg:
                guidance_lines.append(f"- {gg}")

    if verdict_text:
        reasoning_text = verdict_text
        if guidance_lines:
            reasoning_text += "\n\nExecution Notes:\n" + "\n".join(guidance_lines)
    else:
        reasoning_text = "Execution Notes:\n" + "\n".join(guidance_lines) if guidance_lines else "No narrative was returned. (Backend fallback summary applied.)"

    analysis["decision_tier"] = decision_tier
    analysis["decision_label"] = decision_label
    analysis["score"] = score
    analysis["confidence"] = confidence
    analysis["hard_blocks"] = hard_blocks[:10]
    analysis["conditions"] = conditions[:12]
    analysis["rationale"] = rationale[:10]
    analysis["reasoning_text"] = reasoning_text
    analysis["decision"] = decision_label

    return analysis


# ==========================
# Analyze endpoint (trial OR paid)
# ==========================
@app.route("/api/analyze", methods=["POST", "GET"])
def analyze():
    # Allow GET ?test=1 for a clean rate-limit + headers test
    if request.method == "GET":
        if request.args.get("test") == "1":
            ip = _client_ip()
            client_id = _get_client_id_header() or ("ip:" + ip)
            paid_active = _is_client_unlocked(client_id)
            identity_key = f"cid:{client_id}"
            ok, retry_after, rl_headers = _rate_check(identity_key, is_paid=paid_active)
            resp = jsonify({"ok": ok, "mode": "rate_limit_test", "redis_ok": _redis_ping() if _redis_ok() else False})
            resp.status_code = 200 if ok else 429
            for k, v in rl_headers.items():
                resp.headers[k] = v
            return resp
        return jsonify({"ok": False, "error": "Use POST /api/analyze for analysis or GET /api/analyze?test=1"}), 400

    ip = _client_ip()
    client_id = _get_client_id_header() or ("ip:" + ip)

    token = (request.headers.get("X-FXCO-Auth") or "").strip()
    token_payload = _verify_token(token) if token else None

    trial_active = False
    trial_ends_epoch = 0
    token_email = None

    if token_payload and isinstance(token_payload, dict):
        token_email = str(token_payload.get("email") or "").strip().lower() or None
        email = token_email or ""
        trial_ends_epoch = int(token_payload.get("trial_ends") or 0)
        active, te = _trial_active(email)
        trial_active = active
        trial_ends_epoch = te or trial_ends_epoch

    paid_active = _is_client_unlocked(client_id)

    if _require_payment() and not paid_active and not trial_active:
        return jsonify({"error": "Access locked. Start free trial (verify email) or unlock via Pricing."}), 402

    identity_key = f"cid:{client_id}"
    ok, retry_after, rl_headers = _rate_check(identity_key, is_paid=paid_active)
    if not ok:
        resp = jsonify({"error": "Too many requests. Please slow down.", "retry_after_seconds": retry_after})
        resp.status_code = 429
        for k, v in rl_headers.items():
            resp.headers[k] = v
        if trial_ends_epoch:
            resp.headers["X-FXCO-Trial-Ends"] = str(trial_ends_epoch)
        return resp

    pair_type, timeframe, chart_tf_raw, signal_text = _get_payload_fields()

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
                    "timeframe (style)": ["timeframe", "timeframe_mode", "timeframeMode"],
                    "chart_tf (optional)": ["chart_tf", "chartTF", "chart_timeframe", "chartTimeframe"],
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

    img_base64 = None
    file = request.files.get("chart_image")
    if file and file.filename:
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    symbol = detect_symbol_from_signal(signal_text, pair_type)

    hold_minutes = _parse_duration_minutes(signal_text)
    mode_norm = _normalize_mode(timeframe)

    chart_tf = _normalize_chart_tf(chart_tf_raw)
    if chart_tf:
        structure_tf = chart_tf
        execution_tf = _pick_execution_tf(structure_tf)
    else:
        structure_tf, execution_tf = _map_horizon_to_tfs(mode_norm, hold_minutes)

    hold_minutes = hold_minutes if hold_minutes is not None else (20 if mode_norm == "scalp" else 135 if mode_norm == "intraday" else 1440)

    assessment_horizon = {
        "timeframe_style": timeframe,
        "chart_tf": chart_tf or None,
        "mode_normalized": mode_norm,
        "hold_minutes": int(hold_minutes),
        "structure_tf": structure_tf,
        "execution_tf": execution_tf,
    }

    live_snapshot = td_price(symbol)

    candles_structure = td_candles(symbol, interval=structure_tf, limit=160)
    struct_info_structure = {"structure": "unclear", "broken": False, "details": ""}
    if candles_structure.get("ok") and candles_structure.get("values"):
        struct_info_structure = structure_from_candles(candles_structure["values"])

    candles_exec = td_candles(symbol, interval=execution_tf, limit=160)
    struct_info_exec = {"structure": "unclear", "broken": False, "details": ""}
    if candles_exec.get("ok") and candles_exec.get("values"):
        struct_info_exec = structure_from_candles(candles_exec["values"])

    if live_snapshot.get("ok"):
        live_context = f"Live price: {live_snapshot.get('price')} ({symbol})"
        if live_snapshot.get("timestamp_utc"):
            live_context += f" | snapshot_utc: {live_snapshot.get('timestamp_utc')} | session: {live_snapshot.get('session')}"
    else:
        live_context = f"Live data error: {live_snapshot.get('error', 'unknown')}"

    schema_hint = """
{
  "bias": "Long|Short|Neutral|Unclear",
  "strength": 0,
  "clarity": 0,
  "signal_check": {
    "direction": "Long|Short|Neutral|Unclear",
    "entry": "number",
    "stop_loss": "number",
    "targets": [number]
  },
  "market_context": {
    "structure": "string",
    "liquidity": "string",
    "momentum": "string",
    "timeframe_alignment": "string"
  },
  "verdict": "string",
  "guidance": ["string","string","string"]
}
""".strip()

    base_prompt = f"""
You are FX CO-PILOT — a trade validation engine.

You do NOT predict price. You evaluate execution quality: structure, liquidity, alignment, and risk plan.

RISK-DESK RULE (NON-NEGOTIABLE):
- The trade's verdict MUST be driven by the PRIMARY STRUCTURE timeframe (Structure TF).
- The Execution TF may affect entry timing quality, but MUST NOT veto an otherwise valid setup by itself.

User Context:
- Pair type: {pair_type}
- Timeframe style (user): {timeframe}
- Chart timeframe (optional): {chart_tf_raw or ""}
- Intended hold duration (parsed): {hold_minutes} minutes
- Structure TF (verdict driver): {structure_tf}
- Execution TF (entry quality only): {execution_tf}
- Raw signal:
\"\"\"{signal_text}\"\"\"

Live Market:
- Symbol: {symbol}
- {live_context}
- Structure snapshot (Live({structure_tf}) heuristic): {struct_info_structure.get('structure')} ({struct_info_structure.get('details')})
- Execution snapshot (Exec({execution_tf}) heuristic): {struct_info_exec.get('structure')} ({struct_info_exec.get('details')})

Return ONLY valid JSON (no markdown) that matches EXACTLY this schema:

{schema_hint}

Rules:
- Output MUST be raw JSON only.
- entry/stop_loss must be numeric-like strings that can be parsed.
- targets must be an array of numeric-like values.
- If uncertain, keep bias Neutral/Unclear and write a cautious verdict.
- verdict must NEVER be empty.
""".strip()

    messages = [
        {"role": "system", "content": "You are FX Co-Pilot. Output ONLY JSON. No markdown."},
        {"role": "user", "content": base_prompt},
    ]

    if img_base64:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the user's chart screenshot. Use it to refine structure/liquidity/alignment."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                ],
            }
        )

    generated_at = _ms_now()

    def _run_model(strict: bool = False) -> str:
        extra = ""
        if strict:
            extra = "\nIMPORTANT: Output must be JSON ONLY. No commentary. No markdown. No backticks.\n"
        completion = oa.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages + ([{"role": "user", "content": extra}] if strict else []),
            temperature=0.2 if not strict else 0.0,
        )
        return completion.choices[0].message.content or ""

    try:
        raw = _run_model(strict=False)

        analysis_obj = _extract_first_json_object(raw)
        if analysis_obj is None:
            raw2 = _run_model(strict=True)
            analysis_obj = _extract_first_json_object(raw2)
        if analysis_obj is None:
            analysis_obj = _repair_json_with_model(oa, raw, schema_hint)

        if analysis_obj is None:
            resp = jsonify({"error": "Model did not return valid JSON (after repair)."})
            resp.status_code = 502
            for k, v in rl_headers.items():
                resp.headers[k] = v
            if trial_ends_epoch:
                resp.headers["X-FXCO-Trial-Ends"] = str(trial_ends_epoch)
            return resp

        sc = analysis_obj.get("signal_check") or {}
        mc = analysis_obj.get("market_context") or {}

        rr = calculate_rr(sc.get("entry"), sc.get("stop_loss"), sc.get("targets"))

        structure_text = (mc.get("structure") or "").strip() or "Structure context not provided."
        structure_text += f" | Live({structure_tf}): {struct_info_structure.get('structure')} ({struct_info_structure.get('details')})"
        structure_text += f" | Exec({execution_tf}): {struct_info_exec.get('structure')} ({struct_info_exec.get('details')})"

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
                "structure": structure_text,
                "structure_broken": bool(struct_info_structure.get("broken")),
                "liquidity": mc.get("liquidity") or "",
                "momentum": mc.get("momentum") or "",
                "timeframe_alignment": mc.get("timeframe_alignment") or "",
            },
            "verdict": analysis_obj.get("verdict") or "No verdict returned.",
            "guidance": analysis_obj.get("guidance") or [],
            "live_snapshot": live_snapshot,
            "assessment_horizon": assessment_horizon,
        }

        analysis["confidence"] = compute_confidence(analysis)
        analysis["why_this_trade"] = build_why_this_trade(analysis)
        analysis["invalidation_warnings"] = build_invalidation_warnings(analysis, live_snapshot=live_snapshot)
        analysis["execution_plan"] = build_execution_plan(analysis)
        analysis = institutional_decision_engine(analysis)

        decision_tier = str(analysis.get("decision_tier") or "").strip().upper()
        hard_blocks = analysis.get("hard_blocks") or []
        is_blocked = bool(hard_blocks) or (decision_tier == "DO_NOT_TRADE")

        report_payload = {
            "pair_type": pair_type,
            "timeframe": timeframe,
            "chart_tf": chart_tf or None,
            "signal_input": signal_text,
            "analysis": analysis,
            "generated_at": generated_at,
        }

        share_id = None
        share_url = None
        if _sb_ok():
            created = _store_share_report(report_payload, client_id=client_id, email=token_email)
            if created:
                share_id = created["id"]
                share_url = f"{_origin_base_url()}/result.html?r={share_id}"
                report_payload["share_id"] = share_id
                report_payload["share_url"] = share_url
                analysis["share_id"] = share_id
                analysis["share_url"] = share_url

        resp = jsonify(
            {
                "blocked": is_blocked,
                "analysis": analysis,
                "mode": "twelvedata_live",
                "pair_type": pair_type,
                "timeframe": timeframe,
                "chart_tf": chart_tf or None,
                "signal_input": signal_text,
                "generated_at": generated_at,
                "share_id": share_id,
                "share_url": share_url,
                "report": report_payload,
            }
        )
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
