from flask import Flask, request, jsonify, send_from_directory
import base64
import json
import re
from openai import OpenAI

app = Flask(__name__)
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


def build_invalidation_warnings(analysis):
    """
    Heuristic invalidation warnings to prevent false confidence.
    We generate warnings based on bias/decision vs structure keywords,
    as well as poor RR / low confidence / missing key fields.
    """
    warnings = []

    bias = (analysis.get("bias") or "Unclear").lower()
    decision = (analysis.get("decision") or "NEUTRAL").upper().strip()
    conf = analysis.get("confidence") or 0

    mc = analysis.get("market_context", {}) or {}
    sc = analysis.get("signal_check", {}) or {}

    structure = (mc.get("structure") or "").lower()
    momentum = (mc.get("momentum") or "").lower()

    # Parse RR safely
    rr = sc.get("rr")
    rr_f = None
    try:
        rr_f = float(rr) if rr is not None else None
    except:
        rr_f = None

    # --- Missing hard requirements
    if not sc.get("entry") or not sc.get("stop_loss") or not sc.get("targets"):
        warnings.append("Missing key levels (entry / stop loss / targets). Signal is not executable without them.")

    # --- RR quality warnings
    if rr_f is not None and rr_f < 1.2:
        warnings.append(f"Low risk-reward (RR ≈ {rr_f}). Consider improving RR or skipping this setup.")

    # --- Confidence warnings
    if decision == "TAKE TRADE" and conf < 50:
        warnings.append(f"Decision is TAKE but confidence is low ({conf}%). Treat as high-risk or wait for confirmation.")

    # --- Structure invalidation vs bias
    # If bias long but structure says bearish/choch down/bos down/breakdown, warn.
    bearish_tokens = ["bearish", "choch down", "bos down", "breakdown", "lower low", "lower highs", "downtrend"]
    bullish_tokens = ["bullish", "choch up", "bos up", "breakout", "higher high", "higher lows", "uptrend"]

    if "long" in bias:
        if any(t in structure for t in bearish_tokens) or ("choch" in structure and "down" in structure):
            warnings.append("Structure may be broken against a LONG bias (CHoCH/BOS bearish). Invalidate long if price breaks key swing low / support.")
        if "bearish" in momentum:
            warnings.append("Momentum is bearish while bias is LONG. Wait for bullish displacement / confirmation.")

    if "short" in bias:
        if any(t in structure for t in bullish_tokens) or ("choch" in structure and "up" in structure):
            warnings.append("Structure may be broken against a SHORT bias (CHoCH/BOS bullish). Invalidate short if price breaks key swing high / resistance.")
        if "bullish" in momentum:
            warnings.append("Momentum is bullish while bias is SHORT. Wait for bearish displacement / confirmation.")

    # --- Extra safety: if decision TAKE but structure is empty or unclear
    if decision == "TAKE TRADE" and (not structure or "unclear" in structure):
        warnings.append("Structure context is unclear. If price does not confirm BOS/CHoCH as expected, invalidate the trade.")

    # Keep concise
    return warnings[:6]


def normalize_analysis(obj):
    out = {
        "bias": (obj.get("bias") or "Unclear"),
        "confidence": obj.get("confidence") or 0,
        "strength": obj.get("strength") or 0,
        "clarity": obj.get("clarity") or 0,
        "signal_check": obj.get("signal_check") or {},
        "market_context": obj.get("market_context") or {},
        "decision": obj.get("decision") or "NEUTRAL",
        "verdict": obj.get("verdict") or "",
        "guidance": obj.get("guidance") or [],
        "why_this_trade": obj.get("why_this_trade") or [],
        "invalidation_warnings": obj.get("invalidation_warnings") or []
    }

    sc = out["signal_check"] if isinstance(out["signal_check"], dict) else {}
    mc = out["market_context"] if isinstance(out["market_context"], dict) else {}

    direction = sc.get("direction") or sc.get("parsed_direction") or ""
    entry = sc.get("entry") or ""
    stop_loss = sc.get("stop_loss") or sc.get("sl") or ""
    targets = sc.get("targets") or sc.get("tp") or []

    sc_norm = {
        "direction": direction,
        "entry": entry,
        "stop_loss": stop_loss,
        "targets": targets
    }
    sc_norm["targets"] = _parse_targets(sc_norm["targets"])

    rr_val = calculate_rr(sc_norm["entry"], sc_norm["stop_loss"], sc_norm["targets"])
    sc_norm["rr"] = rr_val
    out["signal_check"] = sc_norm

    mc_norm = {
        "structure": mc.get("structure") or "",
        "liquidity": mc.get("liquidity") or "",
        "momentum": mc.get("momentum") or "",
        "timeframe_alignment": mc.get("timeframe_alignment") or ""
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

    # Build invalidation warnings (structure broken, etc)
    out["invalidation_warnings"] = build_invalidation_warnings(out)

    return out


# --------------------------
# FX CO-PILOT — ANALYZER
# Returns JSON {analysis: {...}}
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
- If uncertain, choose NEUTRAL.
"""

    messages = [
        {"role": "system", "content": "You are FX Co-Pilot. Output ONLY JSON."},
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


if __name__ == "__main__":
    app.run(debug=True)
