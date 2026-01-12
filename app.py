from flask import Flask, request, jsonify, send_from_directory
import base64
import json
import time
import threading
from openai import OpenAI

app = Flask(__name__)

# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI()

# -----------------------------
# Simple rate limiter (per IP)
# -----------------------------
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 12
_rate_hits = {}
_rate_lock = threading.Lock()


def get_client_ip():
    return request.remote_addr or "unknown"


def rate_limited(ip):
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    with _rate_lock:
        hits = _rate_hits.get(ip, [])
        hits = [t for t in hits if t >= cutoff]
        if len(hits) >= RATE_LIMIT_MAX:
            _rate_hits[ip] = hits
            return True
        hits.append(now)
        _rate_hits[ip] = hits
    return False


# -----------------------------
# Serve frontend
# -----------------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# -----------------------------
# Helpers
# -----------------------------
def clamp_int(v, default=0):
    try:
        v = int(v)
        return max(0, min(100, v))
    except Exception:
        return default


def normalize_analysis(a):
    return {
        "bias": a.get("bias", "Unclear"),
        "confidence": clamp_int(a.get("confidence")),
        "strength": clamp_int(a.get("strength")),
        "clarity": clamp_int(a.get("clarity")),
        "signal_check": a.get("signal_check", {}),
        "market_context": a.get("market_context", {}),
        "decision": a.get("decision", "NEUTRAL"),
        "verdict": a.get("verdict", ""),
        "guidance": a.get("guidance", []),
    }


# -----------------------------
# Tool schema
# -----------------------------
TRADE_ANALYSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "trade_analysis",
        "parameters": {
            "type": "object",
            "required": [
                "bias",
                "confidence",
                "strength",
                "clarity",
                "signal_check",
                "market_context",
                "decision",
                "verdict",
                "guidance"
            ],
            "properties": {
                "bias": {"type": "string"},
                "confidence": {"type": "integer"},
                "strength": {"type": "integer"},
                "clarity": {"type": "integer"},
                "signal_check": {"type": "object"},
                "market_context": {"type": "object"},
                "decision": {"type": "string"},
                "verdict": {"type": "string"},
                "guidance": {"type": "array", "items": {"type": "string"}}
            }
        }
    }
}


# -----------------------------
# ANALYZE ENDPOINT (FIXED)
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    ip = get_client_ip()
    if rate_limited(ip):
        return jsonify({"error": "Too many requests"}), 429

    pair_type = request.form.get("pair_type", "")
    timeframe = request.form.get("timeframe", "")
    signal_text = request.form.get("signal_input", "")

    file = request.files.get("chart_image")
    img_base64 = None

    if file and file.filename:
        img_base64 = base64.b64encode(file.read()).decode("utf-8")

    messages = [
        {
            "role": "system",
            "content": (
                "You are FX CO-PILOT. "
                "You MUST return structured output using the trade_analysis tool."
            ),
        },
        {
            "role": "user",
            "content": f"""
PAIR TYPE: {pair_type}
TIMEFRAME: {timeframe}
SIGNAL:
{signal_text}
"""
        }
    ]

    if img_base64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Chart screenshot provided."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                }
            ]
        })

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=[TRADE_ANALYSIS_TOOL],
        tool_choice={"type": "function", "function": {"name": "trade_analysis"}},
        temperature=0.2
    )

    msg = completion.choices[0].message

    analysis_obj = None
    try:
        if msg.tool_calls:
            args = msg.tool_calls[0].function.arguments
            analysis_obj = json.loads(args)
    except Exception:
        analysis_obj = None

    if not isinstance(analysis_obj, dict):
        return jsonify({"error": "AI did not return structured output"}), 502

    analysis = normalize_analysis(analysis_obj)

    return jsonify({
        "analysis": analysis,
        "confidence": analysis["confidence"],
        "mode": "tool_structured"
    })


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
