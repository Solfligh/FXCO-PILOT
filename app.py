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
    return jsonify({
        "error": err.description or "Request failed."
    }), err.code or 500


@app.errorhandler(Exception)
def handle_unexpected_exception(err: Exception):
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


def _safe_int(v, default=None, minv=0, maxv=100):
    try:
        n = int(v)
        if n < minv:
            return minv
        if n > maxv:
            return maxv
        return n
    except Exception:
        return default


def _normalize_analysis(obj: dict) -> dict:
    """
    Ensure keys exist + clamp numeric fields.
    """
    if not isinstance(obj, dict):
        obj = {}

    obj.setdefault("bias", "Unclear")
    obj["confidence"] = _safe_int(obj.get("confidence"), default=None, minv=0, maxv=100)
    obj["strength"] = _safe_int(obj.get("strength"), default=None, minv=0, maxv=100)
    obj["clarity"] = _safe_int(obj.get("clarity"), default=None, minv=0, maxv=100)

    obj.setdefault("decision", "NEUTRAL")
    obj.setdefault("verdict", "")
    obj.setdefault("guidance", [])

    if not isinstance(obj.get("guidance"), list):
        obj["guidance"] = []

    obj.setdefault("signal_check", {})
    if not isinstance(obj["signal_check"], dict):
        obj["signal_check"] = {}

    obj.setdefault("market_context", {})
    if not isinstance(obj["market_context"], dict):
        obj["market_context"] = {}

    return obj


def _render_text_from_analysis(a: dict) -> str:
    """
    Provide a human-readable fallback string (not required by the new UI, but useful).
    """
    bias = a.get("bias") or "Unclear"
    decision = (a.get("decision") or "NEUTRAL").upper()
    conf = a.get("confidence")
    strength = a.get("strength")
    clarity = a.get("clarity")

    parts = []
    parts.append(f"BIAS: {bias}")
    if conf is not None:
        parts.append(f"CONFIDENCE: {conf}%")
    if strength is not None:
        parts.append(f"STRENGTH: {strength}")
    if clarity is not None:
        parts.append(f"CLARITY: {clarity}")

    sc = a.get("signal_check") or {}
    parts.append("\nSIGNAL CHECK:")
    parts.append(f"- Parsed direction: {sc.get('direction','')}")
    parts.append(f"- Entry: {sc.get('entry','')}")
    parts.append(f"- Stop loss: {sc.get('stop_loss','')}")
    parts.append(f"- Targets: {sc.get('targets','')}")
    parts.append(f"- Approx RR: {sc.get('rr','')}")

    mc = a.get("market_context") or {}
    parts.append("\nMARKET CONTEXT:")
    parts.append(f"- Structure: {mc.get('structure','')}")
    parts.append(f"- Liquidity: {mc.get('liquidity','')}")
    parts.append(f"- Momentum: {mc.get('momentum','')}")
    parts.append(f"- Timeframe alignment: {mc.get('timeframe_alignment','')}")

    parts.append("\nTRADE DECISION:")
    parts.append(decision)

    if a.get("verdict"):
        parts.append("\nVERDICT:")
        parts.append(a["verdict"])

    if a.get("guidance"):
        parts.append("\nGUIDANCE:")
        for g in a["guidance"][:6]:
            parts.append(f"- {str(g)}")

    return "\n".join(parts).strip()


# --------------------------
# FX CO-PILOT — STRICT JSON ANALYZER
# --------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
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

    img_base64 = None
    if file and file.filename:
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # STRICT JSON schema instructions (keep simple, but explicit)
    system = (
        "You are FX CO-PILOT — an institutional-grade trade validation engine.\n"
        "Return ONLY valid JSON. No markdown. No extra text.\n"
        "All numeric scores are integers 0-100.\n"
        "decision must be exactly one of: TAKE TRADE, NEUTRAL, AVOID TRADE."
    )

    user_prompt = {
        "pair_type": pair_type,
        "timeframe_mode": timeframe,
        "raw_signal": signal_text
    }

    # Required output format (JSON object)
    required_format = {
        "bias": "Long | Short | Neutral | Unclear",
        "confidence": 0,
        "strength": 0,
        "clarity": 0,
        "signal_check": {
            "direction": "",
            "entry": "",
            "stop_loss": "",
            "targets": "",
            "rr": ""
        },
        "market_context": {
            "structure": "",
            "liquidity": "",
            "momentum": "",
            "timeframe_alignment": ""
        },
        "decision": "TAKE TRADE | NEUTRAL | AVOID TRADE",
        "verdict": "",
        "guidance": ["", "", ""]
    }

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Analyze the user's signal and return a JSON object matching this structure.\n"
                f"USER_CONTEXT={json.dumps(user_prompt, ensure_ascii=False)}\n"
                f"REQUIRED_FORMAT={json.dumps(required_format, ensure_ascii=False)}\n"
                "Rules:\n"
                "- If some fields are missing, keep them as empty strings.\n"
                "- Always include all top-level keys.\n"
                "- decision must be one of: TAKE TRADE, NEUTRAL, AVOID TRADE.\n"
                "- confidence/strength/clarity must be integers 0-100.\n"
            )
        }
    ]

    if img_base64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Chart screenshot provided. Use it to refine structure/liquidity/trend/probability."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        })

    # Call model with STRICT JSON output
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    raw_json = completion.choices[0].message.content or "{}"

    # Parse JSON safely
    try:
        parsed = json.loads(raw_json)
    except Exception:
        # As a fallback, still return the raw content (frontend can show it)
        return jsonify({
            "result": raw_json,
            "analysis": None,
            "confidence": None
        })

    analysis = _normalize_analysis(parsed)
    result_text = _render_text_from_analysis(analysis)

    return jsonify({
        "analysis": analysis,
        "confidence": analysis.get("confidence"),
        "result": result_text  # friendly fallback for older UI
    })


if __name__ == "__main__":
    # In production: debug=False
    app.run(debug=True)
