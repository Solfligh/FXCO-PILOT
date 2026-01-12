from flask import Flask, request, jsonify, send_from_directory
import base64
from openai import OpenAI
import re
import time
import threading
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# --------------------------
# HARD LIMITS (UPLOAD / REQUEST SIZE)
# --------------------------
# Total request size limit (includes fields + file).
# If you upload big charts, increase this (e.g. 4 * 1024 * 1024).
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB

# Initialize OpenAI client using environment variable OPENAI_API_KEY
client = OpenAI()


# --------------------------
# SIMPLE IN-MEMORY RATE LIMITER (PER IP)
# --------------------------
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 12

_rate_lock = threading.Lock()
_rate_hits = {}  # ip -> list[timestamps]


def _get_client_ip() -> str:
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        ip = xff.split(",")[0].strip()
        if ip:
            return ip
    return request.remote_addr or "unknown"


def _rate_limited(ip: str) -> bool:
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_lock:
        hits = _rate_hits.get(ip, [])
        hits = [t for t in hits if t >= cutoff]

        if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
            _rate_hits[ip] = hits
            return True

        hits.append(now)
        _rate_hits[ip] = hits

        # small cleanup
        if len(_rate_hits) > 5000:
            stale_cutoff = now - (RATE_LIMIT_WINDOW_SECONDS * 5)
            for k in list(_rate_hits.keys()):
                if not _rate_hits[k] or _rate_hits[k][-1] < stale_cutoff:
                    _rate_hits.pop(k, None)

    return False


# --------------------------
# JSON ERROR HANDLING (CRITICAL)
# --------------------------
@app.errorhandler(413)
def payload_too_large(_err):
    return jsonify({
        "error": "Upload too large. Please upload a smaller image (max ~2MB total request)."
    }), 413


@app.errorhandler(HTTPException)
def handle_http_exception(err: HTTPException):
    """
    Ensures 404/405/etc return JSON instead of HTML.
    """
    return jsonify({
        "error": err.description or "Request failed."
    }), err.code or 500


@app.errorhandler(Exception)
def handle_unexpected_exception(err: Exception):
    """
    Ensures ANY unexpected error returns JSON (prevents HTML debug page breaking fetch JSON).
    """
    return jsonify({
        "error": f"Server error: {str(err)}"
    }), 500


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
# FX CO-PILOT — PRO ANALYZER
# --------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Rate-limit early
        ip = _get_client_ip()
        if _rate_limited(ip):
            return jsonify({
                "error": f"Too many requests. Please wait and try again (limit: {RATE_LIMIT_MAX_REQUESTS} per {RATE_LIMIT_WINDOW_SECONDS}s)."
            }), 429

        pair_type = request.form.get("pair_type", "").strip()
        timeframe = request.form.get("timeframe", "").strip()
        signal_text = request.form.get("signal_input", "").strip()

        file = request.files.get("chart_image")

        if not signal_text and not (file and file.filename):
            return jsonify({"error": "Please paste a signal or upload a chart image."}), 400

        # Optional image -> base64
        img_base64 = None
        if file and file.filename:
            img_bytes = file.read()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        base_prompt = f"""
You are FX CO-PILOT — an institutional-grade trade validation engine operating in PRO MODE.

User Context:
- Pair type: {pair_type}
- Timeframe mode: {timeframe}
- Raw signal (may be unstructured):
\"\"\"{signal_text}\"\"\"

Your Tasks:
1. Parse the full signal:
   - Instrument
   - Direction (Long / Short / Neutral / Unclear)
   - Entry or entry zone
   - Stop loss
   - Take profit levels
   - RR estimation

2. Analyze market logic:
   - Structure (HH, LL, CHoCH, BOS)
   - Liquidity zones (equal highs/lows, sweep zones)
   - Momentum & volatility
   - Timeframe alignment

3. If a chart screenshot is provided, use it to refine:
   - Trend
   - Liquidity grabs
   - Break-of-structure
   - High-probability zones

4. Produce a final decision:
   TAKE TRADE / NEUTRAL / AVOID TRADE

5. Output Format (follow exactly):

BIAS: <Long | Short | Neutral | Unclear>
CONFIDENCE: <0-100%>
STRENGTH: <0-100>
CLARITY: <0-100>

SIGNAL CHECK:
- Parsed direction: ...
- Entry: ...
- Stop loss: ...
- Targets: ...
- Approx RR: ...

MARKET CONTEXT:
- Structure:
- Liquidity:
- Momentum:
- Timeframe alignment:

TRADE DECISION:
<One of: TAKE TRADE / NEUTRAL / AVOID TRADE>

VERDICT:
<1 sentence summary>

GUIDANCE:
- Tip 1
- Tip 2
- Tip 3
"""

        messages = [
            {"role": "system", "content": "You are FX Co-Pilot, an expert AI trade validator operating in institutional PRO MODE."},
            {"role": "user", "content": base_prompt},
        ]

        if img_base64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the user's chart screenshot. Use it to refine structure, liquidity, trend, and probability."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            })

        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.2
        )

        answer = completion.choices[0].message.content or ""

        confidence = None
        match = re.search(r"CONFIDENCE\s*:\s*(\d{1,3})\s*%?", answer, re.IGNORECASE)
        if match:
            confidence = int(match.group(1))
            confidence = max(0, min(confidence, 100))

        return jsonify({
            "result": answer,
            "confidence": confidence
        })

    except Exception as e:
        # Belt + suspenders: ensure /analyze ALWAYS returns JSON
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    # In production: debug=False
    app.run(debug=True)
