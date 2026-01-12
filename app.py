from flask import Flask, request, jsonify, send_from_directory
import base64
from openai import OpenAI
import os

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


# --------------------------
# FX CO-PILOT — PRO ANALYZER
# --------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    # Get text fields
    pair_type = request.form.get("pair_type", "").strip()
    timeframe = request.form.get("timeframe", "").strip()
    signal_text = request.form.get("signal_input", "").strip()

    # Handle uploaded chart image (optional)
    img_base64 = None
    file = request.files.get("chart_image")
    if file and file.filename:
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

    # Construct messages list
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

    # ---------------- Add image if provided (new correct OpenAI format) ----------------
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

    # ---------------- Send to OpenAI ----------------
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.2
        )

        answer = completion.choices[0].message.content
        return jsonify({"result": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# --------------------------
# Run local server
# --------------------------

if __name__ == "__main__":
    app.run(debug=True)
