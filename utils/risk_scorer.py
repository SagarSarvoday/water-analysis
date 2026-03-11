"""
risk_scorer.py
--------------
Calculates a Water Quality Risk Score (0–100)
and WHO-standard parameter safety analysis.
"""

# WHO / standard safe ranges for each parameter
SAFE_RANGES = {
    "ph":               (6.5,  8.5),
    "hardness":         (0,    300),
    "solids":           (0,  500),
    "chloramines":      (0,    4.0),
    "sulfate":          (0,    250),
    "conductivity":     (0,    400),
    "organic_carbon":   (0,    2.0),
    "turbidity":        (0,    1.0),
    "trihalomethanes":  (0,    80),
}

PARAM_LABELS = {
    "ph":               "pH",
    "hardness":         "Hardness (mg/L)",
    "solids":           "Total Dissolved Solids (ppm)",
    "chloramines":      "Chloramines (ppm)",
    "sulfate":          "Sulfate (mg/L)",
    "conductivity":     "Conductivity (μS/cm)",
    "organic_carbon":   "Organic Carbon (ppm)",
    "turbidity":        "Turbidity (NTU)",
    "trihalomethanes":  "Trihalomethanes (μg/L)",
}

PARAM_DESCRIPTIONS = {
    "ph":               "Measures acidity/alkalinity. Safe range: 6.5–8.5.",
    "hardness":         "Calcium & magnesium concentration. High hardness causes scaling.",
    "solids":           "Total dissolved solids. Very high levels affect taste and health.",
    "chloramines":      "Disinfectant used in water treatment. Excess causes irritation.",
    "sulfate":          "Naturally occurring mineral. Excess causes digestive issues.",
    "conductivity":     "Indicates dissolved ion concentration.",
    "organic_carbon":   "Organic compound indicator. High levels signal contamination.",
    "turbidity":        "Water cloudiness. High turbidity signals microbial contamination.",
    "trihalomethanes":  "Disinfection by-product. Excess is potentially carcinogenic.",
}


def analyze_parameters(inputs: dict) -> list[dict]:
    """
    For each parameter, check if it's within the safe range.
    Returns a list of dicts with status info.
    """
    results = []
    for key, value in inputs.items():
        low, high = SAFE_RANGES.get(key, (None, None))
        if low is None:
            continue

        if low <= value <= high:
            status = "Safe"
            color  = "green"
        else:
            status = "Unsafe"
            color  = "red"

        results.append({
            "key":         key,
            "label":       PARAM_LABELS.get(key, key),
            "value":       value,
            "safe_low":    low,
            "safe_high":   high,
            "status":      status,
            "color":       color,
            "description": PARAM_DESCRIPTIONS.get(key, ""),
        })
    return results


def calculate_risk_score(probability_unsafe: float, param_analysis: list[dict]) -> dict:
    """
    Combines ML confidence with parameter violations to produce
    a 0–100 risk score (0 = perfectly safe, 100 = extremely unsafe).
    """
    # ML component: 60% weight
    ml_score = probability_unsafe * 60

    # Parameter violation component: 40% weight
    total_params   = len(param_analysis)
    unsafe_count   = sum(1 for p in param_analysis if p["status"] == "Unsafe")
    param_score    = (unsafe_count / total_params) * 40 if total_params else 0

    risk_score = round(ml_score + param_score, 1)
    risk_score = min(risk_score, 100)

    if risk_score <= 30:
        level = "Low Risk"
        badge = "🟢"
    elif risk_score <= 60:
        level = "Moderate Risk"
        badge = "🟡"
    else:
        level = "High Risk"
        badge = "🔴"

    return {
        "score":       risk_score,
        "level":       level,
        "badge":       badge,
        "unsafe_params": unsafe_count,
        "total_params":  total_params,
    }
