from flask import Flask, request, jsonify, send_from_directory
import base64
from openai import OpenAI
import re
import time
import threading

app = Flask(__name__)

# --------------------------
# HARD LIMITS (UPLOAD / REQUEST SIZE)
# --------------------------
# Total request size limit (includes form fields + file).
# Adjust if needed: 2 * 1024 * 1024 = 2MB
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB

# Initialize OpenAI client using environment variable OPENAI_API_KEY
client = OpenAI()


# --------------------------
# SIMPLE IN-MEMORY RATE LIMITER (PER IP)
# --------------------------
# NOTE:
# - This is best for single-server deployments.
# - If you scale to multiple workers/instances, use Redis-based rate limiting instead.

RATE_LIMIT_WINDOW_SECONDS = 60  # window length
RATE_LIMIT_MAX_REQUESTS = 12    # max /analyze requests per IP per window

_rate_lock = threading.Lock()
_rate_hits = {}  # ip -> list[timestamps]


def _get_client_ip() -> str:
    """
    Best-effort client IP extraction.
    If behind a reverse proxy, ensure it forwards X-Forwarded-For properly.
    """
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        # XFF can be a list: "client, proxy1, proxy2"
        ip = xff.split(",")[0].strip()
        if ip:
            return ip
    return request.remote_addr or "unknown"


def _rate_limited(ip: str) -> bool:
    """
    Sliding-window limiter: keep timestamps within window; block if too many.
    """
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_lock:
        hits = _rate_hits.get(ip, [])

        # keep only recent hits
        hits = [t for t in hits if t >= cutoff]

        if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
            _rate_hits[ip] = hits
            return True

        hits.append(now)
        _rate_hits[ip] = hits

        # opportunistic cleanup to prevent unbounded growth
        # remove stale IP buckets occasionally
        if len(_rate_hits) > 5000:
            stale_cutoff = now - (RATE_LIMIT_WINDOW_SECONDS * 5)
            for k in list(_rate_hits.keys()):
                if not _rate_hits[k] or _rate_hits[k][-1] < stale_cutoff:
                    _rate_hits.pop(k, None)

    return False


# --------------------------
# ERROR HANDLERS
# --------------------------
@app.errorhandler(413)
def payload_too_large(_err):
    return jsonify({
        "error": "Upload too large. Please upload a smaller image (max ~2MB total request)."
    }), 413


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
    # Rate-limit early (cheap)
    ip = _get_client_ip()
    if _rate_limited(ip):
        return jsonify({
            "error": f"Too many requests. Please wait and try again (limit: {RATE_LIMIT_MAX_REQUESTS} per {RATE_LIMIT_WINDOW_SECONDS}s)."
        }), 429

    # Get text fields
    pair_type = request.form.get("pair_type", "").strip()
    timeframe = request.form.get("timeframe", "").strip()
    signal_text = request.form.get("signal_input", "").strip()

    # Basic input sanity (optional but helpful)
    if not signal_text and not request.files.get("chart_image"):
        return jsonify({"error": "Please paste a signal or upload a chart image."}), 400

    # Handle uploaded chart image (optional)
    img_base64 = None
    file = request.files.get("chart_image")

    if file and file.filename:
        # Additional file-size safety: if file stream reports size-like behavior,
        # MAX_CONTENT_LENGTH already blocks oversized total requests.
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # ---------------- PRO MODE PROMPT ----------------
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
        {
            "role": "system",
            "content": "You are FX Co-Pilot, an expert AI trade validator operating in institutional PRO MODE."
        },
        {
            "role": "user",
            "content": base_prompt
        }
    ]

    # Add image if provided (OpenAI vision format)
    if img_base64:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the user's chart screenshot. Use it to refine structure, liquidity, trend, and probability."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        })

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.2
        )

        answer = completion.choices[0].message.content or ""

        # Extract CONFIDENCE from model output
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
        return jsonify({"error": str(e)}), 500


# --------------------------
# Run local server
# --------------------------
if __name__ == "__main__":
    # IMPORTANT: turn debug off in production
    app.run(debug=True)
