import base64
import json
import os
import re
import time
from collections import deque

import requests
from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)

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
# Rate limiting (in-memory)
#   - 5 requests / 60s per IP
#   - 50 requests / 24h per IP
# Applied ONLY to /api/analyze
# ==========================
_SHORT_WINDOW_SECONDS = 60
_SHORT_LIMIT = 5

_DAY_WINDOW_SECONDS = 24 * 60 * 60
_DAY_LIMIT = 50

# key: ip -> deque[timestamps]
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


def _rate_check(ip: str):
    """
    Returns: (ok: bool, retry_after_seconds: int, headers: dict)
    """
    now = time.time()

    if ip not in _RATE_SHORT:
        _RATE_SHORT[ip] = deque()
    if ip not in _RATE_DAY:
        _RATE_DAY[ip] = deque()

    short_q = _RATE_SHORT[ip]
    day_q = _RATE_DAY[ip]

    # prune old entries
    while short_q and (now - short_q[0]) >= _SHORT_WINDOW_SECONDS:
        short_q.popleft()
    while day_q and (now - day_q[0]) >= _DAY_WINDOW_SECONDS:
        day_q.popleft()

    short_remaining = max(0, _SHORT_LIMIT - len(short_q))
    day_remaining = max(0, _DAY_LIMIT - len(day_q))

    # blocked
    if short_remaining <= 0 or day_remaining <= 0:
        retry_after = 1
        if short_remaining <= 0 and short_q:
            retry_after = max(retry_after, int(_SHORT_WINDOW_SECONDS - (now - short_q[0])) + 1)
        if day_remaining <= 0 and day_q:
            retry_after = max(retry_after, int(_DAY_WINDOW_SECONDS - (now - day_q[0])) + 1)

        headers = {
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit-60s": str(_SHORT_LIMIT),
            "X-RateLimit-Remaining-60s": str(short_remaining),
            "X-RateLimit-Limit-24h": str(_DAY_LIMIT),
            "X-RateLimit-Remaining-24h": str(day_remaining),
        }
        return False, retry_after, headers

    # allow: record
    short_q.append(now)
    day_q.append(now)

    headers = {
        "X-RateLimit-Limit-60s": str(_SHORT_LIMIT),
        "X-RateLimit-Remaining-60s": str(short_remaining - 1),
        "X-RateLimit-Limit-24h": str(_DAY_LIMIT),
        "X-RateLimit-Remaining-24h": str(day_remaining - 1),
    }
    return True, 0, headers


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
# Root + health
# ==========================
@app.route("/", methods=["GET"])
def index():
    # backend root is not your frontend (frontend is on solflightech.org)
    return jsonify({"ok": True, "service": "fxco-pilot-backend", "hint": "Use /health and /api/analyze"}), 200


@app.get("/health")
def health():
    return jsonify({"ok": True}), 200


# ==========================
# Helpers
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
    """
    Accepts:
      - multipart/form-data (FormData)
      - application/x-www-form-urlencoded
      - application/json

    Also tolerates frontend key variants.
    Returns: (pair_type, timeframe, signal_text)
    """
    # JSON
    if request.is_json:
        data = request.get_json(silent=True) or {}
        pair_type = _pick_first(data, ["pair_type", "pairType"])
        timeframe = _pick_first(data, ["timeframe", "timeframe_mode", "timeframeMode"])
        signal_text = _pick_first(data, ["signal_input", "signal", "signalText"])
        return pair_type, timeframe, signal_text

    # FormData / urlencoded
    form = request.form or {}
    pair_type = _pick_first(form, ["pair_type", "pairType"])
    timeframe = _pick_first(form, ["timeframe", "timeframe_mode", "timeframeMode"])
    signal_text = _pick_first(form, ["signal_input", "signal", "signalText"])
    return pair_type, timeframe, signal_text


# ==========================
# Analyze endpoint
# ==========================
@app.route("/api/analyze", methods=["POST"])
def analyze():
    # Rate limit ONLY this endpoint
    ip = _client_ip()
    ok, retry_after, rl_headers = _rate_check(ip)
    if not ok:
        resp = jsonify(
            {
                "error": "Too many requests. Please slow down.",
                "limits": {"per_60s": _SHORT_LIMIT, "per_24h": _DAY_LIMIT},
                "retry_after_seconds": retry_after,
            }
        )
        resp.status_code = 429
        for k, v in rl_headers.items():
            resp.headers[k] = v
        return resp

    # IMPORTANT: test mode must happen BEFORE OpenAI + BEFORE validation
    if request.args.get("test") == "1":
        resp = jsonify({"ok": True, "mode": "rate_limit_test", "note": "No OpenAI call made."})
        resp.status_code = 200
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
        # keep OpenAI billing/quota separate from rate limiting
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
