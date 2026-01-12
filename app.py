from flask import Flask, request, jsonify, send_from_directory
import base64
import os
import json
import re
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client using environment variable OPENAI_API_KEY
client = OpenAI()


# --------------------------
# Serve Frontend Files
# --------------------------

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/terms")
def terms():
    # Optional: if you later add a terms.html file
    return send_from_directory(".", "terms.html")


@app.route("/privacy")
def privacy():
    # Optional: if you later add a privacy.html file
    return send_from_directory(".", "privacy.html")


# --------------------------
# Helpers
# --------------------------

def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", "")
    # extract first float-like number
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except:
        return None


def _parse_targets(val):
    """
    Accept:
      - list of numbers/strings
      - string "2050" or "2050, 2060" or "TP1 2050 TP2 2060"
    Returns list[float]
    """
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
    nums = re.findall(r"-?\d+(\.\d+)?", s)
    # re.findall with groups returns tuples sometimes; use regex without groups
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except:
            pass
    return out


def calculate_rr(entry, stop, targets):
    """
    RR = Reward / Risk
    Risk = |Entry - Stop|
    Reward = |Target1 - Entry|
    """
    try:
        entry_f = _to_float(entry)
        stop_f = _to_float(stop)
        targets_f = _parse_targets(targets)

        if entry_f is None or stop_f is None or not targets_f:
            return None

        risk = abs(entry_f - stop_f)
        reward = abs(targets_f[0] - entry_f)

        if risk <= 0:
            return None

        return round(reward / risk, 2)
    except:
        return None


def compute_confidence(analysis):
    """
    Deterministic, explainable score (0-100).
    Weights:
      Structure 30
      RR 25
      Liquidity 20
      Momentum 15
      Completeness 10
    """
    score = 0

    mc = analysis.get("market_context", {}) or {}
    sc = analysis.get("signal_check", {}) or {}

    # 1) Structure (0–30)
    structure = (mc.get("structure") or "").lower()
    if "bos" in structure or "choch" in structure or "break" in structure:
        score += 30
    elif "bullish" in structure or "bearish" in structure:
        score += 28
    elif structure:
        score += 18

    # 2) RR (0–25)
    rr = sc.get("rr")
    try:
        rr_f = float(rr)
        if rr_f >= 2.5:
            score += 25
        elif rr_f >= 2.0:
            score += 20
        elif rr_f >= 1.5:
            score += 12
        elif rr_f >= 1.2:
            score += 6
    except:
        pass

    # 3) Liquidity (0–20)
    liquidity = (mc.get("liquidity") or "").lower()
    if "liquidity" in liquidity or "sweep" in liquidity or "grab" in liquidity or "equal" in liquidity:
        score += 20
    elif "support" in liquidity or "resistance" in liquidity or "zone" in liquidity:
        score += 14
    elif liquidity:
        score += 10

    # 4) Momentum (0–15)
    momentum = (mc.get("momentum") or "").lower()
    if "strong" in momentum:
        score += 15
    elif "bullish" in momentum or "bearish" in momentum:
        score += 10
    elif momentum:
        score += 6

    # 5) Signal completeness (0–10)
    if sc.get("entry") and sc.get("stop_loss") and sc.get("targets"):
        score += 10
    elif sc.get("entry") and sc.get("stop_loss"):
        score += 6

    return min(100, max(0, score))


def build_trade_reasoning(analysis):
    reasons = []

    mc = analysis.get("market_context", {}) or {}
    sc = analysis.get("signal_check", {}) or {}

    if mc.get("structure"):
        reasons.append(f"Market structure supports the setup ({mc.get('structure')}).")

    if mc.get("liquidity"):
        reasons.append(f"Entry aligns with liquidity behavior ({mc.get('liquidity')}).")

    if mc.get("momentum"):
        reasons.append(f"Momentum favors the direction ({mc.get('momentum')}).")

    rr = sc.get("rr")
    try:
        rr_f = float(rr) if rr is not None else None
        if rr_f is not None and rr_f >= 2.0:
            reasons.append(f"Risk-reward is favorable (RR ≈ {rr_f}).")
        elif rr_f is not None:
            reasons.append(f"Risk-reward is acceptable (RR ≈ {rr_f}).")
    except:
        pass

    if sc.get("entry") and sc.get("stop_loss"):
        reasons.append("Defined entry and stop loss reduce execution ambiguity.")

    return reasons[:5]


def normalize_analysis(obj):
    """
    Force the analysis into a clean predictable shape the frontend expects.
    """
    out = {
        "bias": (obj.get("bias") or obj.get("BIAS") or "Unclear"),
        "confidence": obj.get("confidence") or obj.get("CONFIDENCE") or 0,
        "strength": obj.get("strength") or obj.get("STRENGTH") or 0,
        "clarity": obj.get("clarity") or obj.get("CLARITY") or 0,
        "signal_check": obj.get("signal_check") or obj.get("SIGNAL CHECK") or {},
        "market_context": obj.get("market_context") or obj.get("MARKET CONTEXT") or {},
        "decision": obj.get("decision") or obj.get("TRADE DECISION") or "NEUTRAL",
        "verdict": obj.get("verdict") or obj.get("VERDICT") or "",
        "guidance": obj.get("guidance") or obj.get("GUIDANCE") or [],
        "why_this_trade": obj.get("why_this_trade") or []
    }

    # Normalize signal_check fields
    sc = out["signal_check"] if isinstance(out["signal_check"], dict) else {}
    mc = out["market_context"] if isinstance(out["market_context"], dict) else {}

    # Map possible alt keys
    direction = sc.get("direction") or sc.get("parsed_direction") or sc.get("Parsed direction") or sc.get("Parsed direction:")
    entry = sc.get("entry") or sc.get("Entry")
    stop_loss = sc.get("stop_loss") or sc.get("sl") or sc.get("Stop loss") or sc.get("Stop Loss")
    targets = sc.get("targets") or sc.get("tp") or sc.get("Targets")

    sc_norm = {
        "direction": direction or "",
        "entry": entry or "",
        "stop_loss": stop_loss or "",
        "targets": targets or []
    }

    # Ensure targets is list
    sc_norm["targets"] = _parse_targets(sc_norm["targets"])

    # Calculate RR
    rr_val = calculate_rr(sc_norm["entry"], sc_norm["stop_loss"], sc_norm["targets"])
    sc_norm["rr"] = rr_val

    out["signal_check"] = sc_norm

    # Normalize market context keys
    mc_norm = {
        "structure": mc.get("structure") or mc.get("Structure") or "",
        "liquidity": mc.get("liquidity") or mc.get("Liquidity") or "",
        "momentum": mc.get("momentum") or mc.get("Momentum") or "",
        "timeframe_alignment": mc.get("timeframe_alignment") or mc.get("Timeframe alignment") or ""
    }
    out["market_context"] = mc_norm

    # Normalize guidance to list[str]
    g = out.get("guidance")
    if isinstance(g, str):
        out["guidance"] = [x.strip("-• \n\r\t") for x in g.split("\n") if x.strip()]
    elif isinstance(g, list):
        out["guidance"] = [str(x) for x in g if str(x).strip()]
    else:
        out["guidance"] = []

    # Normalize decision
    d = str(out.get("decision") or "NEUTRAL").upper().strip()
    if "TAKE" in d:
        out["decision"] = "TAKE TRADE"
    elif "AVOID" in d:
        out["decision"] = "AVOID TRADE"
    else:
        out["decision"] = "NEUTRAL"

    # Deterministic confidence overrides AI-provided confidence
    out["confidence"] = compute_confidence(out)

    # Build explainer
    out["why_this_trade"] = build_trade_reasoning(out)

    return out


# --------------------------
# FX CO-PILOT — ANALYZER
# Always returns JSON {analysis: {...}}
# --------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    pair_type = request.form.get("pair_type", "").strip()
    timeframe = request.form.get("timeframe", "").strip()
    signal_text = request.form.get("signal_input", "").strip()

    img_base64 = None
    file = request.files.get("chart_image")
    if file and file.filename:
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    base_prompt = f"""
You are FX CO-PILOT — an institutional-grade trade validation engine.

User Context:
- Pair type: {pair_type}
- Timeframe mode: {timeframe}
- Raw signal:
\"\"\"{signal_text}\"\"\"

Return ONLY valid JSON that matches this schema:

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
- Do NOT include markdown.
- Do NOT include extra keys.
- entry/stop_loss/targets MUST be numeric-like.
"""

    messages = [
        {"role": "system", "content": "You are FX Co-Pilot, an expert AI trade validator. Output ONLY JSON."},
        {"role": "user", "content": base_prompt}
    ]

    if img_base64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the chart screenshot. Use it to refine structure/liquidity/trend."},
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

        # Attempt JSON parse (model instructed to return JSON only)
        try:
            analysis_obj = json.loads(raw)
        except Exception:
            return jsonify({
                "error": "Model did not return valid JSON.",
                "debug_preview": raw[:400]
            }), 502

        if not isinstance(analysis_obj, dict):
            return jsonify({"error": "Model JSON was not an object."}), 502

        analysis = normalize_analysis(analysis_obj)

        return jsonify({
            "analysis": analysis,
            "confidence": analysis.get("confidence", 0),
            "mode": "json_only"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------
# Run local server
# --------------------------

if __name__ == "__main__":
    app.run(debug=True)
