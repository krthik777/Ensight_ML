"""
services/nilm_engine.py
------------------------
Smart 3-tier NILM engine.

Tier 1 — DB Appliance Matching (most accurate when user has registered appliances):
    Greedy power matching against the user's own registered appliances in MongoDB.
    Strict power conservation: sum of detected cannot exceed mains total.
    Stops adding appliances once residual power < 5% of total.

Tier 2 — LSTM Model with power-conservation post-filter:
    Falls back to trained LSTM. After detection, removes appliances whose
    combined predicted power exceeds the actual mains reading.

Tier 3 — Heuristic Rules:
    Last resort when no readings exist. Returns empty if mean_power = 0.
"""

import numpy as np
from datetime import datetime


# ── Appliance type → canonical name ──────────────────────────────────────────
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
    raw_type = appliance.get("type", "")
    raw_name = appliance.get("name", "")
    return _TYPE_TO_NAME.get(raw_type, raw_name.lower().replace(" ", "_"))


# ════════════════════════════════════════════════════════════════════════════
# Tier 1 — DB Appliance Matching
# ════════════════════════════════════════════════════════════════════════════

def match_appliances_to_power(
    mean_power_w: float,
    db_appliances: list,
) -> dict:
    """
    Greedy matching: find which registered appliances best explain the mains reading.

    Rules:
      1. Sort appliances by rated power (highest first).
      2. Mark an appliance ON if remaining unexplained power ≥ 70% of its rated power.
         (70% threshold — must account for most of its typical load, not just noise)
      3. Stop early when remaining power < 5% of total (further additions are noise).
      4. Final guard: sum of detected appliance powers must not exceed total × 1.05.

    An appliance with powerRating = 0 or estimatedWattage = 0 is always skipped
    because we have no credible power reference for it.
    """
    if not db_appliances or mean_power_w <= 30:
        # < 30W total — too low to reliably detect anything
        return _empty_result("db_matching")

    # Minimum threshold to stop adding more appliances.
    # Once <5% of total power is unexplained, further detections are noise.
    stop_threshold = max(30.0, mean_power_w * 0.05)

    sorted_apps = sorted(
        db_appliances,
        key=lambda a: (a.get("estimatedWattage") or a.get("powerRating") or 0),
        reverse=True,
    )

    remaining_w   = mean_power_w
    active        = []
    confidence    = {}
    power_breakdown = {}

    for app in sorted_apps:
        # Stop when we've explained ≥ 95% of total power
        if remaining_w < stop_threshold:
            break

        typical_w = app.get("estimatedWattage") or app.get("powerRating") or 0
        if typical_w <= 0:
            # No credible power reference — skip
            continue

        # ON condition: remaining unexplained power must be ≥ 70% of rated power.
        # This prevents detecting e.g. a 1500W AC when only 200W is unexplained.
        if remaining_w >= typical_w * 0.70:
            name    = _canonical(app)
            consumed = min(typical_w, remaining_w)

            # Confidence based on how well the appliance fits the remaining load
            # (consumed / typical): 1.0 = perfect fit, 0.7 = minimum
            fit_ratio  = consumed / typical_w
            conf       = round(max(0.60, min(0.99, fit_ratio)), 2)

            active.append(name)
            power_breakdown[name] = round(consumed, 2)
            confidence[name]      = conf
            remaining_w           = max(0.0, remaining_w - consumed)

    total_matched = mean_power_w - remaining_w

    print(
        f"🔍 [nilm_engine] DB matching → "
        f"{len(active)}/{len(db_appliances)} appliances active | "
        f"mains={mean_power_w:.0f}W  matched={total_matched:.0f}W  "
        f"unmatched={remaining_w:.0f}W"
    )

    return {
        "active_appliances": active,
        "confidence":        confidence,
        "power_breakdown":   power_breakdown,
        "total_matched_w":   round(total_matched, 2),
        "unmatched_w":       round(remaining_w, 2),
        "tier":              "db_matching",
    }


# ════════════════════════════════════════════════════════════════════════════
# Tier 2 — LSTM Model with power-conservation filter
# ════════════════════════════════════════════════════════════════════════════

def run_lstm_detection(
    mains_sequence: list,
    mean_power_w: float,
    inference_service,
) -> dict:
    """
    Run the LSTM model and apply a power-conservation post-filter.

    The LSTM often over-predicts (outputs non-zero for appliances that are off).
    Post-filter: remove appliances with lowest confidence until
    sum(active powers) ≤ total_mains × 1.10.

    Also enforces a minimum power threshold per appliance:
    an appliance must be predicted to consume ≥ max(15W, 2% of mains) to count as ON.
    """
    raw  = inference_service.detect_appliances(mains_sequence)
    pwr  = inference_service.predict_appliance_power(mains_sequence)

    active_raw = raw.get("active_appliances", [])
    conf_raw   = raw.get("confidence", {})
    power_map  = pwr.get("appliances", {})

    # --- Minimum per-appliance power threshold ---
    # An appliance must account for at least 2% of mains or 15W (whichever larger)
    min_watts = max(15.0, mean_power_w * 0.02)

    active = [
        a for a in active_raw
        if power_map.get(a, 0) >= min_watts
    ]
    conf   = {a: conf_raw.get(a, 0.5) for a in active}
    power  = {a: round(power_map.get(a, 0), 2) for a in active}

    # --- Power conservation: prune until sum ≤ mains × 1.10 ---
    if mean_power_w > 0:
        budget = mean_power_w * 1.10
        # Sort by confidence ascending so we remove least-confident first
        while sum(power.get(a, 0) for a in active) > budget and active:
            # Remove lowest confidence appliance
            worst = min(active, key=lambda a: conf.get(a, 0))
            active.remove(worst)
            conf.pop(worst, None)
            power.pop(worst, None)

    total = sum(power.values())

    print(
        f"🤖 [nilm_engine] LSTM → {len(active)} active "
        f"(after power-conservation filter, budget={mean_power_w:.0f}W)"
    )

    return {
        "active_appliances": active,
        "confidence":        conf,
        "power_breakdown":   power,
        "total_matched_w":   round(total, 2),
        "unmatched_w":       round(max(0, mean_power_w - total), 2),
        "tier":              "lstm_model",
    }


# ════════════════════════════════════════════════════════════════════════════
# Tier 3 — Heuristic fallback
# ════════════════════════════════════════════════════════════════════════════

# (name, min_power_threshold_w, typical_w)
_HEURISTIC_RULES = [
    ("air_conditioner",  900,  1500),
    ("washing_machine",  350,   500),
    ("water_motor",      300,   600),
    ("iron",             700,  1000),
    ("kitchen_outlets",  200,   400),
    ("television",       100,   150),
    ("laptop_computer",   50,    65),
    ("fridge",            80,   120),
    ("water_filter",      30,    45),
]


def heuristic_detection(mean_power_w: float) -> dict:
    """
    Rule-based fallback — deterministic (no randomness).
    An appliance is ON if the total mains reading exceeds its minimum threshold
    AND adding it does not cause the total to exceed mains power.
    Returns empty if mean_power_w < 30W (no meaningful load).
    """
    if mean_power_w < 30:
        return _empty_result("heuristic")

    remaining     = mean_power_w
    active        = []
    confidence    = {}
    power_breakdown = {}
    stop_threshold = max(30.0, mean_power_w * 0.05)

    for name, threshold, typical in _HEURISTIC_RULES:
        if remaining < stop_threshold:
            break
        if mean_power_w > threshold and remaining >= typical * 0.70:
            consumed = min(typical, remaining)
            active.append(name)
            power_breakdown[name] = round(consumed, 2)
            confidence[name]      = round(min(0.85, consumed / typical), 2)
            remaining -= consumed

    print(
        f"⚙️  [nilm_engine] Heuristic → {len(active)} active "
        f"(mains={mean_power_w:.0f}W)"
    )

    return {
        "active_appliances": active,
        "confidence":        confidence,
        "power_breakdown":   power_breakdown,
        "total_matched_w":   round(mean_power_w - remaining, 2),
        "unmatched_w":       round(remaining, 2),
        "tier":              "heuristic",
    }


# ════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════

def smart_nilm_detect(
    mains_sequence:   list,
    db_appliances:    list,
    inference_service,
) -> dict:
    """
    3-tier NILM detection with power-conservation guarantee.

    Returns consistent dict with:
      active_appliances, confidence, power_breakdown,
      total_power_w, mean_power_w, unmatched_w, tier, timestamp
    """
    mean_power_w = float(np.mean(mains_sequence)) if mains_sequence else 0.0

    # ── Tier 1: user's registered appliances ─────────────────────────────────
    if db_appliances:
        result = match_appliances_to_power(mean_power_w, db_appliances)
    # ── Tier 2: LSTM model ────────────────────────────────────────────────────
    elif mains_sequence and inference_service and inference_service.model is not None:
        result = run_lstm_detection(mains_sequence, mean_power_w, inference_service)
    # ── Tier 3: heuristic ─────────────────────────────────────────────────────
    else:
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
