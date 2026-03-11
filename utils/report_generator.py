"""
report_generator.py
-------------------
Uses Groq API to generate a natural language
water quality report based on ML prediction + parameter analysis.
"""

from groq import Groq


def _build_prompt(prediction, probability, inputs, param_analysis, risk):
    status = "SAFE FOR DRINKING" if prediction == 1 else "NOT SAFE FOR DRINKING"
    confidence = round(probability * 100, 1)

    unsafe_params = [p for p in param_analysis if p["status"] == "Unsafe"]
    safe_params   = [p for p in param_analysis if p["status"] == "Safe"]

    unsafe_text = "\n".join(
        f"  - {p['label']}: {p['value']} (safe range: {p['safe_low']}–{p['safe_high']})"
        for p in unsafe_params
    ) or "  None"

    safe_text = ", ".join(p["label"] for p in safe_params) or "None"
    param_block = "\n".join(f"  {p['label']}: {p['value']}" for p in param_analysis)

    return f"""You are a water quality expert AI assistant.
A water sample has been analyzed using a machine learning model.
Generate a detailed professional water quality report.

--- ANALYSIS DATA ---
Overall Prediction  : {status}
Model Confidence    : {confidence}%
Risk Score          : {risk['score']} / 100
Risk Level          : {risk['level']}

Water Parameters Measured:
{param_block}

Parameters OUTSIDE safe limits:
{unsafe_text}

Parameters within safe limits: {safe_text}

--- REPORT INSTRUCTIONS ---
Write the report in these exact sections:
1. Executive Summary
2. Parameter-by-Parameter Analysis
3. Key Health Risks
4. Treatment Recommendations
5. Conclusion

Do not use # markdown headers. Use plain section titles.
Keep total length around 300-400 words."""


def generate_report(prediction, probability, inputs, param_analysis, risk, api_key):
    client = Groq(api_key=api_key)

    prompt = _build_prompt(prediction, probability, inputs, param_analysis, risk)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    return response.choices[0].message.content


def generate_assistant_response(question, prediction, probability, inputs,
                                 param_analysis, risk, chat_history, api_key):
    client = Groq(api_key=api_key)

    status = "SAFE FOR DRINKING" if prediction == 1 else "NOT SAFE FOR DRINKING"
    confidence = round(probability * 100, 1)
    param_block = "\n".join(
        f"  {p['label']}: {p['value']} — {p['status']}" for p in param_analysis
    )

    system_prompt = f"""You are an expert water quality AI assistant.
You help users understand the results of a water quality analysis.

Current water sample context:
- Prediction    : {status}
- Confidence    : {confidence}%
- Risk Score    : {risk['score']} / 100 ({risk['level']})
- Parameters:
{param_block}

Answer clearly and concisely. Keep response under 150 words."""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=400,
    )

    return response.choices[0].message.content
