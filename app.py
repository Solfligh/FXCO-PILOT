from flask import Flask, request, jsonify, send_from_directory
import base64
from openai import OpenAI
import time
import threading
from werkzeug.exceptions import HTTPException
import json

app = Flask(__name__)

# --------------------------
# HARD LIMITS (UPLOAD / REQUEST SIZE)
# --------------------------
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB total request

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
    return False


# --------------------------
# JSON ERROR HANDLING
# --------------------------
@app.errorhandler(413)
def payload_too_large(_err):
    return jsonify({"error": "Upload too large. Please upload a smaller image (max ~2MB total request)."}), 413


@app.errorhandler(HTTPException)
def handle_http_exception(err: HTTPException):
    return jsonify({"error": err.description or "Request failed."}), err.code or 500


@app.errorhandler(Exception)
def handle_unexpected_exception(err: Exception):
    return jsonify({"error": f"Server error: {str(err)}"}), 500


# --------------------------
# Serve Frontend Files
# --------------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


def _clamp_int(v, default=0, minv=0, maxv=100):
    try:
        n = int(v)
        return max(minv, min(maxv, n))
    except Exception:
        return default


def _normalize_analysis(obj: dict) -> dict:
    if not isinstance(obj, dict):
        obj = {}

    bias = obj.get("bias") or "Unclear"
    if bias not in ["Long", "Short", "Neutral", "Unclear"]:
        bias = "Unclear"

    decision = (obj.get("decision") or "NEUTRAL").upper()
    if decision not in ["TAKE TRADE", "NEUTRAL", "AVOID TRADE"]:
        decision = "NEUTRAL"

    sc = obj.get("signal_check") if isinstance(obj.get("signal_check"), dict) else {}
    mc = obj.get("market_context") if isinstance(obj.get("market_context"), dict) else {}

    guidance = obj.get("guidance") if isinstance(obj.get("guidance"), list) else []
    guidance = [str(x) for x in guidance if str(x).strip()][:6]

    return {
        "bias": bias,
        "confidence": _clamp_int(obj.get("confidence"), default=0),
        "strength": _clamp_int(obj.get("strength"), default=0),
        "clarity": _clamp_int(obj.get("clarity"), default=0),
        "signal_check": {
            "direction": str(sc.get("direction", "") or ""),
            "entry": str(sc.get("entry", "") or ""),
            "stop_loss": str(sc.get("stop_loss", "") or ""),
            "targets": str(sc.get("targets", "") or ""),
            "rr": str(sc.get("rr", "") or "")
        },
        "market_context": {
            "structure": str(mc.get("structure", "") or ""),
            "liquidity": str(mc.get("liquidity", "") or ""),
            "momentum": str(mc.get("momentum", "") or ""),
            "timeframe_alignment": str(mc.get("timeframe_alignment", "") or "")
        },
        "decision": decision,
        "verdict": str(obj.get("verdict", "") or ""),
        "guidance": guidance
    }


TRADE_ANALYSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "trade_analysis",
        "description": "Return structured trade analysis for FXCO-PILOT UI.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "bias": {"type": "string", "enum": ["Long", "Short", "Neutral", "Unclear"]},
                "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                "strength": {"type": "integer", "minimum": 0, "maximum": 100},
                "clarity": {"type": "integer", "minimum": 0, "maximum": 100},
                "signal_check": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "direction": {"type": "string"},
                        "entry": {"type": "string"},
                        "stop_loss": {"type": "string"},
                        "targets": {"type": "string"},
                        "rr": {"type": "string"}
                    },
                    "required": ["direction", "entry", "stop_loss", "targets", "rr"]
                },
                "market_context": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "structure": {"type": "string"},
                        "liquidity": {"type": "string"},
                        "momentum": {"type": "string"},
                        "timeframe_alignment": {"type": "string"}
                    },
                    "required": ["structure", "liquidity", "momentum", "timeframe_alignment"]
                },
                "decision": {"type": "string", "enum": ["TAKE TRADE", "NEUTRAL", "AVOID TRADE"]},
                "verdict": {"type": "string"},
                "guidance": {"type": "array", "items": {"type": "string"}, "maxItems": 6}
            },
            "required": ["bias", "confidence", "strength", "clarity", "signal_check", "market_context", "decision", "verdict", "guidance"]
        }
    }
}


@app.route("/analyze", methods=["POST"])
def analyze():

    
    ip = _get_client_ip()
    if _rate_limited(ip):
        return jsonify({"error": f"Too many requests. Please wait and try again (limit: {RATE_LIMIT_MAX_REQUESTS} per {RATE_LIMIT_WINDOW_SECONDS}s)."}), 429

    pair_type = request.form.get("pair_type", "").strip()
    timeframe = request.form.get("timeframe", "").strip()
    signal_text = request.form.get("signal_input", "").strip()

    file = request.files.get("chart_image")
    if not signal_text and not (file and file.filename):
        return jsonify({"error": "Please paste a signal or upload a chart image."}), 400

    img_base64 = None
    if file and file.filename:
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    system = (
        "You are FX CO-PILOT — an institutional-grade trade validation engine.\n"
        "You MUST call the tool `trade_analysis` with your structured result.\n"
        "Rules:\n"
        "- decision must be exactly one of: TAKE TRADE, NEUTRAL, AVOID TRADE.\n"
        "- confidence/strength/clarity are integers 0-100.\n"
        "- Keep entries concise and practical.\n"
        "- If chart is missing, clearly state assumptions in structure/liquidity fields."
    )

    user_text = (
        f"PAIR_TYPE: {pair_type}\n"
        f"TIMEFRAME_MODE: {timeframe}\n"
        f"RAW_SIGNAL:\n{signal_text}\n\n"
        "Return a complete structured result using the tool call."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]

    if img_base64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Chart screenshot provided. Use it to refine structure/liquidity/trend/probability."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        })

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=[TRADE_ANALYSIS_TOOL],
        tool_choice={"type": "function", "function": {"name": "trade_analysis"}},
        temperature=0.2,
    )

    msg = completion.choices[0].message

    analysis_obj = None
    tool_args_raw = None

    try:
        if msg.tool_calls and len(msg.tool_calls) > 0:
            tc = msg.tool_calls[0]
            tool_args_raw = tc.function.arguments  # JSON string
            analysis_obj = json.loads(tool_args_raw)
    except Exception:
        analysis_obj = None

    if not isinstance(analysis_obj, dict):
        # IMPORTANT: return a JSON error, not plain text
        return jsonify({
            "error": "AI did not return structured output. Please retry.",
            "debug_tool_args": tool_args_raw,
            "debug_msg_content": (msg.content or "")
        }), 502

    analysis = _normalize_analysis(analysis_obj)

    # ✅ ALWAYS return analysis so frontend uses renderAnalysisFromJSON
  
return jsonify({
        "analysis": analysis,
        "confidence": analysis.get("confidence", 0),
        "mode": "tool_structured"
    })


if __name__ == "__main__":
    app.run(debug=True)
