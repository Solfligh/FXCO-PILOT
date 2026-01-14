from flask import Flask, request, jsonify, send_from_directory
import base64
import os
import re
import json
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ==========================
# Load .env (ENV VAR ONLY)
# ==========================
load_dotenv()

app = Flask(__name__)

# ==========================
# OpenAI client (uses OPENAI_API_KEY from env automatically)
# ==========================
client = OpenAI()

# ==========================
# Twelve Data (Grow) - ENV VAR ONLY
# ==========================
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TD_BASE = "https://api.twelvedata.com"

# ==========================
# Public safety: simple rate limiting + caching (in-memory)
# ==========================
CACHE = {}  # key -> {"ts": float, "data": any}
CACHE_TTL_SECONDS = 12

RATE = {}  # ip -> {"window_start": float, "count": int}
RATE_WINDOW_SECONDS = 60
RATE_MAX_REQUESTS_PER_WINDOW = 90  # tune later


def _client_ip() -> str:
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_ok(ip: str) -> bool:
    now = time.time()
    rec = RATE.get(ip)
    if not rec:
        RATE[ip] = {"window_start": now, "count": 1}
        return True

    if now - rec["window_start"] >= RATE_WINDOW_SECONDS:
        RATE[ip] = {"window_start": now, "count": 1}
        return True

    rec["count"] += 1
    return rec["count"] <= RATE_MAX_REQUESTS_PER_WINDOW


def _cache_get(key: str):
    rec = CACHE.get(key)
    if not rec:
        return None
    if time.time() - rec["ts"] > CACHE_TTL_SECONDS:
        return None
    return rec["data"]


def _cache_set(key: str, data):
    CACHE[key] = {"ts": time.time(), "data": data}


# --------------------------
# Serve Frontend Files
# --------------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# --------------------------
# FAVICON ROUTES (prevents /favicon.ico 404)
# --------------------------
@app.route("/favicon.ico")
def favicon_ico():
    return send_from_directory("static", "favicon.ico")


# Optional: if you have these files, this prevents noisy 404s from browsers/devices
@app.route("/favicon-32.png")
def favicon_32():
    return send_from_directory("static", "favicon-32.png")


@app.route("/favicon-16.png")
def favicon_16():
    return send_from_directory("static", "favicon-16.png")


@app.route("/apple-touch-icon.png")
def apple_touch_icon():
    return send_from_directory("static", "apple-touch-icon.png")


# ==========================
# Health (new)
# ==========================
@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "FXCO-PILOT",
        "twelvedata_key_loaded": bool(TWELVE_DATA_API_KEY),
        "time": datetime.utcnow().isoformat() + "Z"
    })


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
    except:
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
        except:
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
            timeout=10
        )
        data = r.json()
        if "status" in data and data["status"] == "error":
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
                "apikey": TWELVE_DATA_API_KEY
            },
            timeout=12
        )
        data = r.json()
        if "status" in data and data["status"] == "error":
            out = {"ok": False, "error": data.get("message", "Twelve Data error")}
            _cache_set(cache_key, out)
            return out

        values = data.get("values") or []
        # values are returned newest-first typically
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


# ==========================
# NEW: API candles endpoint (for live charts)
# ==========================
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
        candles.append({
            "datetime": v.get("datetime"),
            "open": v.get("open"),
            "high": v.get("high"),
            "low": v.get("low"),
            "close": v.get("close"),
        })

    return jsonify({
        "ok": True,
        "symbol": symbol,
        "interval": interval,
        "candles": candles,  # newest-first
        "source": "twelvedata"
    })


# ==========================
# Trend / structure detection (simple + safe)
# ==========================
def structure_from_candles(values):
    """
    values: list of dicts from Twelve Data time_series
    We'll classify structure as bullish/bearish/unclear and detect "structure broken".
    Simple heuristic:
      - bullish if last close > close N and last swing low is higher than earlier swing low
      - bearish if last close < close N and last swing high is lower than earlier swing high
      - broken if opposite break happens relative to implied bias levels
    """
    if not values or len(values) < 40:
        return {"structure": "unclear", "broken": False, "details": "Not enough candle data."}

    # Convert to oldest->newest for analysis
    vals = list(reversed(values))

    try:
        closes = [float(v["close"]) for v in vals]
        highs = [float(v["high"]) for v in vals]
        lows = [float(v["low"]) for v in vals]
    except Exception:
        return {"structure": "unclear", "broken": False, "details": "Candle parse error."}

    last = closes[-1]
    prev = closes[-25]  # ~25 bars back
    trend = "bullish" if last > prev else "bearish" if last < prev else "unclear"

    # crude swing points
    recent_low = min(lows[-15:])
    prior_low = min(lows[-40:-15])
    recent_high = max(highs[-15:])
    prior_high = max(highs[-40:-15])

    if trend == "bullish":
        # bullish structure: higher low + pushing highs
        if recent_low > prior_low:
            return {"structure": "bullish", "broken": False, "details": "Higher low detected."}
        # if trend bullish but recent_low <= prior_low, structure weakening
        return {"structure": "unclear", "broken": False, "details": "Bullish trend but HL not confirmed."}

    if trend == "bearish":
        # bearish structure: lower high + pushing lows
        if recent_high < prior_high:
            return {"structure": "bearish", "broken": False, "details": "Lower high detected."}
        return {"structure": "unclear", "broken": False, "details": "Bearish trend but LH not confirmed."}

    return {"structure": "unclear", "broken": False, "details": "Trend unclear."}


def decide_block(bias: str, struct: str, live_price: float, entry: float, sl: float):
    """
    BLOCK rules (strict):
      - If SL already hit (based on live price) => BLOCK
      - If bias long and structure bearish => BLOCK
      - If bias short and structure bullish => BLOCK
    """
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


# ==========================
# Deterministic confidence / Why / Warnings
# ==========================
def compute_confidence(analysis):
    score = 0
    mc = analysis.get("market_context", {}) or {}
    sc = analysis.get("signal_check", {}) or {}

    struct = (mc.get("structure") or "").lower()
    if struct in ["bullish", "bearish"]:
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
    if "liquidity" in liquidity or "sweep" in liquidity or "grab" in liquidity or "equal" in liquidity:
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

    # Missing levels
    if not sc.get("entry") or not sc.get("stop_loss") or not sc.get("targets"):
        warnings.append("Missing key levels (entry / SL / targets).")

    rr = sc.get("rr")
    if isinstance(rr, (int, float)) and rr < 1.2:
        warnings.append(f"Low RR (≈ {rr}). Consider skipping or improving RR.")

    # Live invalidation
    if live_snapshot and live_snapshot.get("ok"):
        price = live_snapshot.get("price")
        sl = _to_float(sc.get("stop_loss"))
        if sl is not None:
            if "long" in bias and price <= sl:
                warnings.append("Live price has hit SL => invalidated.")
            if "short" in bias and price >= sl:
                warnings.append("Live price has hit SL => invalidated.")

    # Structure mismatch warning
    if ("long" in bias and struct == "bearish") or ("short" in bias and struct == "bullish"):
        warnings.append("Structure is against your bias (structure broken).")

    return warnings[:8]


# ==========================
# NEW: Simple deterministic signal endpoint
# ==========================
def _ema(series, period: int):
    if not series or period <= 1:
        return series[:]
    k = 2 / (period + 1)
    out = []
    prev = series[0]
    out.append(prev)
    for v in series[1:]:
        prev = (v * k) + (prev * (1 - k))
        out.append(prev)
    return out


def _rsi(series, period: int = 14):
    if len(series) < period + 1:
        return [50.0] * len(series)

    gains, losses = [], []
    for i in range(1, len(series)):
        diff = series[i] - series[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    out = [50.0] * period
    out.append(100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out.append(100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))))

    if len(out) < len(series):
        out = [50.0] * (len(series) - len(out)) + out
    return out[:len(series)]


def _atr(highs, lows, closes, period: int = 14):
    if len(closes) < 2:
        return [0.0] * len(closes)

    tr = []
    for i in range(len(closes)):
        if i == 0:
            tr.append(highs[i] - lows[i])
        else:
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))

    out = [0.0] * len(closes)
    if len(tr) < period:
        return out

    first = sum(tr[:period]) / period
    out[period - 1] = first
    prev = first
    for i in range(period, len(tr)):
        prev = ((prev * (period - 1)) + tr[i]) / period
        out[i] = prev
    for i in range(period - 1):
        out[i] = out[period - 1]
    return out


def _build_signal(values):
    # values newest-first -> analyze oldest->newest
    vals = list(reversed(values))
    closes = [float(v["close"]) for v in vals]
    highs = [float(v["high"]) for v in vals]
    lows = [float(v["low"]) for v in vals]

    e20 = _ema(closes, 20)
    e50 = _ema(closes, 50)
    r = _rsi(closes, 14)
    a = _atr(highs, lows, closes, 14)

    last_close = closes[-1]
    last_e20 = e20[-1]
    last_e50 = e50[-1]
    last_rsi = r[-1]
    last_atr = a[-1] if a[-1] > 0 else (last_close * 0.001)

    trend_up = last_e20 > last_e50
    trend_down = last_e20 < last_e50

    direction = "WAIT"
    confidence = 50
    reasons = []

    if trend_up:
        reasons.append("Trend up (EMA20 > EMA50).")
        confidence += 12
    elif trend_down:
        reasons.append("Trend down (EMA20 < EMA50).")
        confidence += 12
    else:
        reasons.append("Trend flat (EMA20 ~ EMA50).")
        confidence -= 10

    if trend_up:
        if last_rsi < 45:
            reasons.append(f"RSI {last_rsi:.1f} low in uptrend (pullback entry).")
            confidence += 15
            direction = "BUY"
        elif 45 <= last_rsi <= 65:
            reasons.append(f"RSI {last_rsi:.1f} supportive in uptrend.")
            confidence += 8
            direction = "BUY"
        else:
            reasons.append(f"RSI {last_rsi:.1f} high (pullback risk).")
            confidence -= 12
            direction = "WAIT"

    if trend_down:
        if last_rsi > 55:
            reasons.append(f"RSI {last_rsi:.1f} high in downtrend (pullback entry).")
            confidence += 15
            direction = "SELL"
        elif 35 <= last_rsi <= 55:
            reasons.append(f"RSI {last_rsi:.1f} supportive in downtrend.")
            confidence += 8
            direction = "SELL"
        else:
            reasons.append(f"RSI {last_rsi:.1f} low (bounce risk).")
            confidence -= 12
            direction = "WAIT"

    confidence = max(0, min(100, confidence))

    rr_target = 2.0
    if confidence < 55:
        rr_target = 1.5
    if confidence < 45:
        rr_target = 1.2

    entry = last_close
    sl_dist = 1.2 * last_atr

    if direction == "BUY":
        sl = entry - sl_dist
        tp = entry + sl_dist * rr_target
        invalidation = "Structure broken if price closes below SL zone."
    elif direction == "SELL":
        sl = entry + sl_dist
        tp = entry - sl_dist * rr_target
        invalidation = "Structure broken if price closes above SL zone."
    else:
        sl = None
        tp = None
        invalidation = "No active trade. Waiting for higher-quality setup."

    warnings = []
    if direction in ("BUY", "SELL") and confidence < 55:
        warnings.append("Low-confidence setup: consider waiting for confirmation.")
    if last_atr / entry > 0.01:
        warnings.append("High volatility detected (ATR elevated): reduce position size.")

    return {
        "direction": direction,
        "confidence": confidence,
        "rr_target": rr_target,
        "entry": round(entry, 6),
        "sl": round(sl, 6) if sl is not None else None,
        "tp": round(tp, 6) if tp is not None else None,
        "atr": round(last_atr, 6),
        "rsi": round(last_rsi, 2),
        "ema20": round(last_e20, 6),
        "ema50": round(last_e50, 6),
        "why": " ".join(reasons),
        "warnings": warnings[:8],
        "invalidation": invalidation,
    }


@app.get("/api/signal")
def api_signal():
    ip = _client_ip()
    if not _rate_ok(ip):
        return jsonify({"ok": False, "error": "Rate limit exceeded. Please slow down."}), 429

    symbol = _norm_symbol(request.args.get("symbol", "") or "EURUSD")
    interval = (request.args.get("interval", "") or "15min").strip()
    limit = int(request.args.get("limit", "") or 180)

    snap = td_candles(symbol, interval=interval, limit=limit)
    if not snap.get("ok"):
        return jsonify(snap), 502

    values = snap.get("values") or []
    if len(values) < 60:
        return jsonify({"ok": False, "error": "Not enough candle data to compute a signal."}), 400

    signal = _build_signal(values)
    live_snapshot = td_price(symbol)

    # strict SL hit check (block)
    blocked = False
    reason = ""
    if live_snapshot.get("ok") and signal.get("direction") in ("BUY", "SELL") and signal.get("sl") is not None:
        p = live_snapshot.get("price")
        if signal["direction"] == "BUY" and p <= signal["sl"]:
            blocked, reason = True, "Live price is at/through Stop Loss. Trade invalidated."
        if signal["direction"] == "SELL" and p >= signal["sl"]:
            blocked, reason = True, "Live price is at/through Stop Loss. Trade invalidated."

    return jsonify({
        "ok": True,
        "blocked": blocked,
        "block_reason": reason,
        "symbol": symbol,
        "interval": interval,
        "live_snapshot": live_snapshot,
        "signal": signal,
        "disclaimer": "Educational use only. Not financial advice. Trading involves risk."
    })


# ==========================
# Main Analyze Endpoint (keeps your existing flow)
# ==========================
@app.route("/analyze", methods=["POST"])
def analyze():
    ip = _client_ip()
    if not _rate_ok(ip):
        return jsonify({"error": "Rate limit exceeded. Please slow down."}), 429

    # Get text fields
    pair_type = request.form.get("pair_type", "").strip()
    timeframe = request.form.get("timeframe", "").strip()
    signal_text = request.form.get("signal_input", "").strip()

    # Chart image optional
    img_base64 = None
    file = request.files.get("chart_image")
    if file and file.filename:
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Detect symbol
    symbol = detect_symbol_from_signal(signal_text, pair_type)

    # ==========================
    # Live data: price + candles
    # ==========================
    live_snapshot = td_price(symbol)
    candles_snapshot = td_candles(symbol, interval="5min", limit=120)

    struct_info = {"structure": "unclear", "broken": False, "details": ""}
    if candles_snapshot.get("ok") and candles_snapshot.get("values"):
        struct_info = structure_from_candles(candles_snapshot["values"])

    # ==========================
    # Ask AI for structured JSON (same UI contract)
    # ==========================
    # Provide live info to AI
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
"""

    messages = [
        {"role": "system", "content": "You are FX Co-Pilot. Output ONLY JSON."},
        {"role": "user", "content": base_prompt}
    ]

    # Add image (if provided)
    if img_base64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the user's chart screenshot. Use it to refine structure/liquidity/trend."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        })

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.2
        )

        raw = completion.choices[0].message.content or ""

        try:
            analysis_obj = json.loads(raw)
        except Exception:
            return jsonify({
                "error": "Model did not return valid JSON.",
                "debug_preview": raw[:500],
                "live_snapshot": live_snapshot
            }), 502

        if not isinstance(analysis_obj, dict):
            return jsonify({
                "error": "Model JSON was not an object.",
                "live_snapshot": live_snapshot
            }), 502

        # ==========================
        # Normalize + RR + deterministic confidence
        # ==========================
        sc = analysis_obj.get("signal_check") or {}
        mc = analysis_obj.get("market_context") or {}

        entry = sc.get("entry")
        sl = sc.get("stop_loss")
        targets = sc.get("targets")

        rr = calculate_rr(entry, sl, targets)

        # Force structure string to include live/candle signal
        mc_struct = mc.get("structure") or ""
        mc["structure"] = (mc_struct + f" | Live(5m): {struct_info.get('structure')}").strip(" |")

        # Build normalized analysis object
        analysis = {
            "bias": analysis_obj.get("bias") or "Unclear",
            "strength": analysis_obj.get("strength") or 0,
            "clarity": analysis_obj.get("clarity") or 0,
            "signal_check": {
                "direction": sc.get("direction") or "Unclear",
                "entry": _to_float(entry),
                "stop_loss": _to_float(sl),
                "targets": _parse_targets(targets),
                "rr": rr
            },
            "market_context": {
                "structure": mc.get("structure") or "",
                "liquidity": mc.get("liquidity") or "",
                "momentum": mc.get("momentum") or "",
                "timeframe_alignment": mc.get("timeframe_alignment") or ""
            },
            "decision": analysis_obj.get("decision") or "NEUTRAL",
            "verdict": analysis_obj.get("verdict") or "",
            "guidance": analysis_obj.get("guidance") or [],
            "live_snapshot": live_snapshot
        }

        # Normalize decision
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

        # ==========================
        # BLOCK trade (strict)
        # ==========================
        live_price = live_snapshot.get("price") if live_snapshot.get("ok") else None
        bias = analysis.get("bias")
        struct = struct_info.get("structure")  # bullish/bearish/unclear
        entry_f = analysis["signal_check"].get("entry")
        sl_f = analysis["signal_check"].get("stop_loss")

        blocked, reason = decide_block(bias, struct, live_price, entry_f, sl_f)

        if blocked:
            # Force decision
            analysis["decision"] = "AVOID TRADE"
            analysis["verdict"] = (analysis.get("verdict") or "").strip()
            analysis["verdict"] = (analysis["verdict"] + " " if analysis["verdict"] else "") + f"TRADE BLOCKED: {reason}"
            analysis["invalidation_warnings"] = [reason] + (analysis.get("invalidation_warnings") or [])

            return jsonify({
                "blocked": True,
                "block_reason": reason,
                "analysis": analysis,
                "mode": "twelvedata_block"
            })

        return jsonify({
            "blocked": False,
            "analysis": analysis,
            "mode": "twelvedata_live"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "FXCO-PILOT",
        "status": "running"
    })


# --------------------------
# Run local server
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
