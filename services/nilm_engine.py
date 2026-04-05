"""
services/nilm_engine.py
------------------------
Smart 3-tier NILM (Non-Intrusive Load Monitoring) engine.

Tier 1 — DB Appliance Matching (most accurate):
    If the user has appliances registered in MongoDB for this room,
    use greedy power-signature matching to determine which ones are ON.
    This uses the user's own powerRating / estimatedWattage data.

Tier 2 — LSTM Model:
    Fall back to the trained LSTM model when no room appliances are found.

Tier 3 — Heuristic Rules:
    If both above fail (model not loaded, no data), use rule-based
    thresholds as a last resort.
"""

import numpy as np
from datetime import datetime

# ── Appliance type → canonical NILM name mapping ─────────────────────────────
_TYPE_TO_NAME = {
    "Air Conditioner":   "air_conditioner",
    "Refrigerator":      "fridge",
    "Television":        "television",
    "Washing Machine":   "washing_machine",
    "Laptop":            "laptop_computer",
    "Computer":          "laptop_computer",
    "Kitchen Appliance": "kitchen_outlets",
    "Iron":              "iron",
    "Water Filter":      "water_filter",
    "Water Pump":        "water_motor",
    "Water Motor":       "water_motor",
    "Other":             "other",
}

def _canonical(appliance: dict) -> str:
    """Return a normalised name key for an appliance."""
    raw_name = appliance.get("name", "")
    raw_type = appliance.get("type", "")
    return _TYPE_TO_NAME.get(raw_type, raw_name.lower().replace(" ", "_"))


# ── Tier 1: DB appliance power matching ──────────────────────────────────────

def match_appliances_to_power(
    mean_power_w: float,
    db_appliances: list[dict],
    tolerance: float = 0.20,         # ±20 % of rated power counts as "possible ON"
) -> dict:
    """
    Greedy matching: which combination of registered appliances best
    explains the observed mains power?

    Algorithm:
      1. Sort appliances by estimatedWattage descending.
      2. Accumulate: mark an appliance ON if adding its load doesn't
         exceed the remaining unexplained power by more than `tolerance`.
      3. Confidence = 1 - (abs_error / mean_power_w).

    Args:
        mean_power_w : Average mains power over last N readings (Watts).
        db_appliances: List of appliance dicts from MongoDB.
        tolerance    : Fractional over-allocation allowed per appliance.

    Returns:
        {
          "active_appliances": [...],
          "confidence":        {name: score, ...},
          "power_breakdown":   {name: watts, ...},
          "total_matched_w":   float,
          "unmatched_w":       float,   # power not explained by known appliances
          "tier":              "db_matching",
        }
    """
    if not db_appliances or mean_power_w <= 0:
        return _empty_result("db_matching")

    # Sort highest power first so large appliances are checked first
    sorted_apps = sorted(
        db_appliances,
        key=lambda a: a.get("estimatedWattage") or a.get("powerRating") or 0,
        reverse=True,
    )

    remaining_w = mean_power_w
    active = []
    confidence = {}
    power_breakdown = {}

    for app in sorted_apps:
        typical_w = app.get("estimatedWattage") or app.get("powerRating") or 0
        if typical_w <= 0:
            continue

        # ON if its load fits within remaining + tolerance
        if remaining_w >= typical_w * (1 - tolerance):
            name = _canonical(app)
            consumed = min(typical_w, remaining_w)
            active.append(name)
            power_breakdown[name] = round(consumed, 2)

            # Confidence: how closely the appliance power fits the remaining load
            fit_ratio = consumed / typical_w
            conf = max(0.5, min(0.99, fit_ratio))
            confidence[name] = round(conf, 2)

            remaining_w = max(0.0, remaining_w - consumed)

    total_matched = mean_power_w - remaining_w

    print(
        f"🔍 [nilm_engine] DB matching → "
        f"{len(active)} active / {len(db_appliances)} registered | "
        f"matched={total_matched:.0f}W, unmatched={remaining_w:.0f}W"
    )

    return {
        "active_appliances": active,
        "confidence":        confidence,
        "power_breakdown":   power_breakdown,
        "total_matched_w":   round(total_matched, 2),
        "unmatched_w":       round(remaining_w,   2),
        "tier":              "db_matching",
    }


# ── Tier 2: LSTM model wrapper ────────────────────────────────────────────────

def run_lstm_detection(
    mains_sequence: list[float],
    inference_service,
) -> dict:
    """
    Call the LSTM-based NILMInferenceService and normalise output to the
    same shape as match_appliances_to_power().
    """
    raw = inference_service.detect_appliances(mains_sequence)
    pwr = inference_service.predict_appliance_power(mains_sequence)

    active = raw.get("active_appliances", [])
    conf   = raw.get("confidence", {})
    power  = pwr.get("appliances", {})

    total = sum(power.get(a, 0) for a in active)

    print(f"🤖 [nilm_engine] LSTM → {len(active)} active appliances detected")

    return {
        "active_appliances": active,
        "confidence":        conf,
        "power_breakdown":   {a: round(power.get(a, 0), 2) for a in active},
        "total_matched_w":   round(total, 2),
        "unmatched_w":       0.0,
        "tier":              "lstm_model",
    }


# ── Tier 3: heuristic fallback ────────────────────────────────────────────────

_HEURISTIC_RULES = [
    # (name,             threshold_w, typical_w, base_prob)
    ("air_conditioner",  900,  1500, 0.50),
    ("washing_machine",  400,   500, 0.20),
    ("water_motor",      350,   600, 0.15),
    ("iron",             700,  1000, 0.10),
    ("kitchen_outlets",  250,   400, 0.35),
    ("television",        80,   150, 0.40),
    ("laptop_computer",   40,    65, 0.45),
    ("fridge",            50,   120, 0.80),   # almost always on
    ("water_filter",      20,    45, 0.70),
]

def heuristic_detection(mean_power_w: float) -> dict:
    """Rule-based fallback — no model, no DB appliances needed."""
    active = []
    confidence = {}
    power_breakdown = {}
    remaining = mean_power_w

    for name, threshold, typical, base_prob in _HEURISTIC_RULES:
        if remaining <= 0:
            break
        is_on = (
            mean_power_w > threshold
            and np.random.random() < base_prob + (mean_power_w / 5000) * 0.2
        )
        if is_on:
            consumed = min(typical, remaining)
            active.append(name)
            power_breakdown[name] = round(consumed, 2)
            confidence[name] = round(base_prob, 2)
            remaining -= consumed

    print(
        f"⚙️  [nilm_engine] Heuristic → {len(active)} active "
        f"(mean={mean_power_w:.0f}W)"
    )

    return {
        "active_appliances": active,
        "confidence":        confidence,
        "power_breakdown":   power_breakdown,
        "total_matched_w":   round(mean_power_w - max(remaining, 0), 2),
        "unmatched_w":       round(max(remaining, 0), 2),
        "tier":              "heuristic",
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def smart_nilm_detect(
    mains_sequence:  list[float],
    db_appliances:   list[dict],
    inference_service,
) -> dict:
    """
    Pick the best available NILM tier and return a normalised result dict.

    Args:
        mains_sequence   : Recent power readings (Watts), oldest → newest.
        db_appliances    : Appliances from MongoDB for this room (may be []).
        inference_service: NILMInferenceService instance (may have None model).

    Returns: dict with keys:
        active_appliances, confidence, power_breakdown,
        total_power_w, unmatched_w, mean_power_w, tier, timestamp
    """
    mean_power_w = float(np.mean(mains_sequence)) if mains_sequence else 0.0

    # ── Tier 1 ───────────────────────────────────────────────────────────────
    if db_appliances:
        result = match_appliances_to_power(mean_power_w, db_appliances)
        result["mean_power_w"]  = round(mean_power_w, 2)
        result["total_power_w"] = round(mean_power_w, 2)
        result["timestamp"]     = datetime.now().isoformat()
        return result

    # ── Tier 2 ───────────────────────────────────────────────────────────────
    if mains_sequence and inference_service and inference_service.model is not None:
        result = run_lstm_detection(mains_sequence, inference_service)
        result["mean_power_w"]  = round(mean_power_w, 2)
        result["total_power_w"] = round(mean_power_w, 2)
        result["timestamp"]     = datetime.now().isoformat()
        return result

    # ── Tier 3 ───────────────────────────────────────────────────────────────
    result = heuristic_detection(mean_power_w)
    result["mean_power_w"]  = round(mean_power_w, 2)
    result["total_power_w"] = round(mean_power_w, 2)
    result["timestamp"]     = datetime.now().isoformat()
    return result


def _empty_result(tier: str) -> dict:
    return {
        "active_appliances": [],
        "confidence":        {},
        "power_breakdown":   {},
        "total_matched_w":   0.0,
        "unmatched_w":       0.0,
        "tier":              tier,
    }
