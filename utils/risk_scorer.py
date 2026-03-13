"""
risk_scorer.py  —  WHO-standard parameter safety analysis and risk scoring.
"""

SAFE_RANGES = {
    "ph":              (6.5,  8.5),
    "hardness":        (0,    500.0),
    "solids":          (0,    600.0),
    "chloramines":     (0,    4.0),
    "sulfate":         (0,    500.0),
    "conductivity":    (0,    400.0),
    "organic_carbon":  (0,    4.0),
    "turbidity":       (0,    4.0),
    "trihalomethanes": (0,    80.0),
}


PARAM_LABELS = {
    "ph":              "pH",
    "hardness":        "Hardness (mg/L)",
    "solids":          "Solids (ppm)",
    "chloramines":     "Chloramines (ppm)",
    "sulfate":         "Sulfate (mg/L)",
    "conductivity":    "Conductivity",
    "organic_carbon":  "Organic Carbon",
    "turbidity":       "Turbidity (NTU)",
    "trihalomethanes": "Trihalomethanes",
}


def analyze_parameters(inputs: dict) -> list:
    results = []
    for key, value in inputs.items():
        low, high = SAFE_RANGES.get(key, (None, None))
        if low is None:
            continue
        status = "Safe" if low <= value <= high else "Unsafe"
        results.append({
            "key":       key,
            "label":     PARAM_LABELS.get(key, key),
            "value":     value,
            "safe_low":  low,
            "safe_high": high,
            "status":    status,
        })
    return results


def calculate_risk_score(probability_unsafe: float, param_analysis: list) -> dict:
    ml_score     = probability_unsafe * 60
    total        = len(param_analysis)
    unsafe_count = sum(1 for p in param_analysis if p["status"] == "Unsafe")
    param_score  = (unsafe_count / total) * 40 if total else 0
    risk_score   = min(round(ml_score + param_score, 1), 100)

    if risk_score <= 30:
        level, badge = "Low Risk",      "🟢"
    elif risk_score <= 60:
        level, badge = "Moderate Risk", "🟡"
    else:
        level, badge = "High Risk",     "🔴"

    return {
        "score":         risk_score,
        "level":         level,
        "badge":         badge,
        "unsafe_params": unsafe_count,
        "total_params":  total,
    }
