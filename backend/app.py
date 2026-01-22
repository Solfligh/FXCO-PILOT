import base64
import json
import os
import re
import time
import requests

from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

app = Flask(__name__)

# ==========================
# Cache Implementation
# ==========================
_CACHE = {}
_CACHE_TTL = {}


def _cache_get(key):
    """Retrieve cached value if not expired."""
    if key in _CACHE and time.time() < _CACHE_TTL.get(key, 0):
        return _CACHE[key]
    return None


def _cache_set(key, value, ttl=60):
    """Cache a value with TTL in seconds."""
    _CACHE[key] = value
    _CACHE_TTL[key] = time.time() + ttl


# ==========================
# Rate Limiting Implementation
# ==========================
_RATE_LIMIT = {}


def _client_ip():
    """
    Get client IP address.

    - If behind a proxy/CDN, X-Forwarded-For may contain a comma-separated list.
      The left-most IP is the original client.
    """
    xff = request.headers.get("X-Forwarded-For", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_ok(ip, limit=30, window=60):
    """Check if IP is within rate limit."""
    now = time.time()
    if ip not in _RATE_LIMIT:
        _RATE_LIMIT[ip] = []
    _RATE_LIMIT[ip] = [t for t in _RATE_LIMIT[ip] if now - t < window]
    if len(_RATE_LIMIT[ip]) >= limit:
        return False
    _RATE_LIMIT[ip].append(now)
    return True


# ==========================
# OpenAI client (LAZY INIT - prevents startup crash if key missing)
# ==========================
def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# ==========================
# Twelve Data - ENV VAR ONLY
# ==========================
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TD_BASE = "https://api.twelvedata.com"


# --------------------------
# Root route (backend is NOT your frontend)
# --------------------------
@app.get("/")
def root():
    # Your frontend lives on Vercel; Render is API only.
    return jsonify(
        {
            "ok": True,
            "service": "fxco-pilot-backend",
            "endpoints": ["/health", "/api/analyze", "/api/candles", "/quote"],
        }
    ), 200


# --------------------------
# Serve optional static files (harmless; returns 204 if missing)
# --------------------------
@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


def _send_static_if_exists(filename: str):
    static_dir = os.path.join(os.getcwd(), "static")
    full_path = os.path.join(static_dir, filename)
    if os.path.isfile(full_path):
        return send_from_directory("static", filename)
    return ("", 204)


@app.route("/favicon.ico")
def favicon_ico():
    return _send_static_if_exists("favicon.ico")


@app.route("/favicon-32.png")
def favicon_32():
    return _send_static_if_exists("favicon-32.png")


@app.route("/favicon-16.png")
def favicon_16():
    return _send_static_if_exists("favicon-16.png")


@app.route("/apple-touch-icon.png")
def apple_touch_icon():
    return _send_static_if_exists("apple-touch-icon.png")


# --------------------------
# Health Check
# --------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True}), 200


# ==========================
# Helpers: symbols / parsing
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
# Twelve Data calls (with cache)
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
            _cache_set(cache_key, out)
            return out

        p = float(data["price"])
        out = {"ok": True, "symbol": symbol, "price": p, "source": "twelvedata"}
        _cache_set(cache_key, out)
        return out
    except Exception as e:
        out = {"ok": False, "error": str(e)}
        _cache_set(cache_key, out)
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
            _cache_set(cache_key, out)
            return out

        values = data.get("values") or []
        out = {"ok": True, "symbol": symbol, "interval": interval, "values": values, "source": "twelvedata"}
        _cache_set(cache_key, out)
        return out
    except Exception as e:
        out = {"ok": False, "error": str(e)}
        _cache_set(cache_key, out)
        return out


@app.route("/quote", methods=["GET"])
def quote():
    ip = _client_ip()
    if not _rate_ok(ip):
        return jsonify({"ok": False, "error": "Rate limit exceeded. Please slow down."}), 429

    symbol = _norm_symbol(request.args.get("symbol", "") or "EURUSD")
    q = td_price(symbol)
    return jsonify(q)


@app.get("/api/candles")
def api_candles():
    ip = _client_ip()
    if not _rate_ok(ip):
        return jsonify({"ok": False, "error": "Rate limit exceeded. Please slow down."}), 429

    symbol = request.args.get("symbol", "EURUSD")
    interval = request.args.get("interval", "5min")
    limit = request.args.get("limit", 120)

    symbol = _norm_symbol(symbol)
    try:
        limit = int(limit)
    except Exception:
        limit = 120

    snap = td_candles(symbol, interval=interval, limit=limit)
    if not snap.get("ok"):
        return jsonify(snap), 502

    values = snap.get("values") or []
    candles = []
    for v in values:
        candles.append(
            {
                "datetime": v.get("datetime"),
                "open": v.get("open"),
                "high": v.get("high"),
                "low": v.get("low"),
                "close": v.get("close"),
            }
        )

    return jsonify(
        {
            "ok": True,
            "symbol": symbol,
            "interval": interval,
            "candles": candles,
            "source": "twelvedata",
        }
    )


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


def decide_block(bias: str, struct: str, live_price: float, entry: float, sl: float):
    b = (bias or "unclear").lower()

    if sl is not None and live_price is not None:
        if "long" in b and live_price <= sl:
            return True, "Live price is at/through Stop Loss. Trade invalidated."
        if "short" in b and live_price >= sl:
            return True, "Live price is at/through Stop Loss. Trade invalidated."

    if "long" in b and struct == "bearish":
        return True, "Structure is bearish while bias is LONG (structure broken)."
    if "short" in b and struct == "bullish":
        return True, "Structure is bullish while bias is SHORT (structure broken)."

    return False, ""


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


def _read_analyze_payload():
    """
    Supports:
    - multipart/form-data (from HTML <form>) with optional file chart_image
    - application/json (from PowerShell / fetch) WITHOUT files
    """
    pair_type = ""
    timeframe = ""
    signal_text = ""
    img_base64 = None

    # JSON request
    if request.is_json:
        data = request.get_json(silent=True) or {}
        pair_type = str(data.get("pair_type", "")).strip()
        timeframe = str(data.get("timeframe", "")).strip()
        signal_text = str(data.get("signal_input", "")).strip()

        # Optional: allow sending base64 chart in JSON
        img_base64 = data.get("chart_image_base64")
        if isinstance(img_base64, str):
            img_base64 = img_base64.strip() or None

        return pair_type, timeframe, signal_text, img_base64

    # multipart/form-data
    pair_type = request.form.get("pair_type", "").strip()
    timeframe = request.form.get("timeframe", "").strip()
    signal_text = request.form.get("signal_input", "").strip()

    file = request.files.get("chart_image")
    if file and file.filename:
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return pair_type, timeframe, signal_text, img_base64


@app.route("/api/analyze", methods=["POST"])
def analyze():
    ip = _client_ip()
    if not _rate_ok(ip):
        return jsonify({"error": "Rate limit exceeded. Please slow down."}), 429

    oa = _get_openai_client()
    if oa is None:
        return jsonify({"error": "Missing OPENAI_API_KEY on server."}), 500

    pair_type, timeframe, signal_text, img_base64 = _read_analyze_payload()

    # You can decide whether to hard-require these.
    # I recommend requiring them so empty requests don't burn API calls.
    if not pair_type or not timeframe or not signal_text:
        return jsonify(
            {
                "error": "Missing required fields.",
                "expected": ["pair_type", "timeframe", "signal_input"],
                "got": {"pair_type": bool(pair_type), "timeframe": bool(timeframe), "signal_input": bool(signal_text)},
            }
        ), 400

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
- Use live price and structure notes to avoid late entries.
- If uncertain, choose NEUTRAL.
""".strip()

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

        if not isinstance(analysis_obj, dict):
            return jsonify({"error": "Model JSON was not an object.", "live_snapshot": live_snapshot}), 502

        sc = analysis_obj.get("signal_check") or {}
        mc = analysis_obj.get("market_context") or {}

        entry = sc.get("entry")
        sl = sc.get("stop_loss")
        targets = sc.get("targets")

        rr = calculate_rr(entry, sl, targets)

        mc_struct = mc.get("structure") or ""
        mc["structure"] = (mc_struct + f" | Live(5m): {struct_info.get('structure')}").strip(" |")

        analysis = {
            "bias": analysis_obj.get("bias") or "Unclear",
            "strength": analysis_obj.get("strength") or 0,
            "clarity": analysis_obj.get("clarity") or 0,
            "signal_check": {
                "direction": sc.get("direction") or "Unclear",
                "entry": _to_float(entry),
                "stop_loss": _to_float(sl),
                "targets": _parse_targets(targets),
                "rr": rr,
            },
            "market_context": {
                "structure": mc.get("structure") or "",
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

        live_price = live_snapshot.get("price") if live_snapshot.get("ok") else None
        bias = analysis.get("bias")
        struct = struct_info.get("structure")
        entry_f = analysis["signal_check"].get("entry")
        sl_f = analysis["signal_check"].get("stop_loss")

        blocked, reason = decide_block(bias, struct, live_price, entry_f, sl_f)

        if blocked:
            analysis["decision"] = "AVOID TRADE"
            analysis["verdict"] = (analysis.get("verdict") or "").strip()
            analysis["verdict"] = (analysis["verdict"] + " " if analysis["verdict"] else "") + f"TRADE BLOCKED: {reason}"
            analysis["invalidation_warnings"] = [reason] + (analysis.get("invalidation_warnings") or [])

            return jsonify({"blocked": True, "block_reason": reason, "analysis": analysis, "mode": "twelvedata_block"})

        return jsonify({"blocked": False, "analysis": analysis, "mode": "twelvedata_live"})

    except json.JSONDecodeError:
        return jsonify({"error": "Model did not return valid JSON."}), 502

    except Exception as e:
        # Try to return correct status for OpenAI quota/auth issues
        msg = str(e)
        status = getattr(e, "status_code", None)

        if status == 401:
            return jsonify({"error": "OpenAI auth failed (bad API key).", "details": msg}), 401
        if status == 429 or "insufficient_quota" in msg or "Error code: 429" in msg:
            return jsonify({"error": "OpenAI quota exceeded / billing issue.", "details": msg}), 429

        return jsonify({"error": msg}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
