"""
services/energy_analytics.py
------------------------------
Accurate energy and cost analytics using real MongoDB readings.

Daily kWh  — trapezoidal integration of power readings over time.
Monthly     — confirmed actual + EWMA-projected remaining days.
KSEB cost   — 2025-2027 telescopic slab tariff with full bill breakdown.
"""

import numpy as np
from datetime import datetime, timezone, timedelta


# ── Daily kWh (trapezoidal integration) ──────────────────────────────────────

def compute_daily_kwh(readings: list[dict]) -> dict:
    """
    Compute kWh for a list of {'power': W, 'timestamp': datetime, 'energy': kWh}
    readings using trapezoidal integration of the power curve.

    Falls back to energy-field delta if cumulative energy is populated.

    Args:
        readings: Sorted list (oldest → newest) from powerreadings collection.
                  Each dict must have 'power' (Watts) and 'timestamp' (datetime).
                  Optionally 'energy' (cumulative kWh).

    Returns:
        {
          "kwh":           float,   # total kWh for the period
          "avg_power_w":   float,   # average watts
          "peak_power_w":  float,   # max watts seen
          "duration_hours":float,   # actual span of readings
          "reading_count": int,
          "method":        str,     # "energy_delta" | "trapezoidal"
        }
    """
    if not readings:
        return _empty_energy()

    # ── Strategy 1: cumulative energy field ──────────────────────────────────
    first_energy = readings[0].get("energy")
    last_energy  = readings[-1].get("energy")
    if (
        first_energy is not None and last_energy is not None
        and last_energy > 0 and last_energy >= first_energy
    ):
        kwh = round(last_energy - first_energy, 4)
        ts_span = _ts_span_hours(readings)
        return {
            "kwh":            kwh,
            "avg_power_w":    round((kwh / ts_span) * 1000, 2) if ts_span > 0 else 0,
            "peak_power_w":   round(max(r["power"] for r in readings), 2),
            "duration_hours": round(ts_span, 3),
            "reading_count":  len(readings),
            "method":         "energy_delta",
        }

    # ── Strategy 2: trapezoidal integration over power readings ───────────────
    total_kwh   = 0.0
    power_vals  = []

    for i in range(1, len(readings)):
        p1 = readings[i - 1]["power"]
        p2 = readings[i]["power"]
        t1 = _to_utc(readings[i - 1]["timestamp"])
        t2 = _to_utc(readings[i]["timestamp"])

        dt_h = (t2 - t1).total_seconds() / 3600.0

        # Sanity: skip if timestamps are out of order or gap > 2 h
        if dt_h <= 0 or dt_h > 2.0:
            continue

        avg_kw    = (p1 + p2) / 2.0 / 1000.0
        total_kwh += avg_kw * dt_h
        power_vals.extend([p1, p2])

    power_vals = power_vals or [r["power"] for r in readings]
    ts_span    = _ts_span_hours(readings)

    return {
        "kwh":            round(total_kwh, 4),
        "avg_power_w":    round(float(np.mean(power_vals)), 2),
        "peak_power_w":   round(float(np.max(power_vals)), 2),
        "duration_hours": round(ts_span, 3),
        "reading_count":  len(readings),
        "method":         "trapezoidal",
    }


# ── Multi-day series ──────────────────────────────────────────────────────────

def compute_daily_series(daily_kwh_map: dict[str, float]) -> dict:
    """
    Given a {date_str: kwh} map, compute basic statistics.
    date_str format: "YYYY-MM-DD"
    """
    if not daily_kwh_map:
        return {"days": [], "avg_daily_kwh": 0.0, "total_kwh": 0.0}

    values = list(daily_kwh_map.values())
    return {
        "days":          [
            {"date": d, "kwh": round(v, 4)}
            for d, v in sorted(daily_kwh_map.items())
        ],
        "avg_daily_kwh": round(float(np.mean(values)), 4),
        "total_kwh":     round(float(sum(values)), 4),
    }


# ── Monthly forecast with EWMA trend ─────────────────────────────────────────

def forecast_month_end(
    daily_kwh_list: list[float],   # kWh for each day collected so far (oldest first)
    days_elapsed:   int,            # how many days into the month
    rate_per_kwh:   float = 6.5,
    monthly_budget_kwh: float = 400.0,
) -> dict:
    """
    Forecast end-of-month kWh and cost from 2+ days of readings.

    Method:
      1. Compute EWMA of available daily kWh (more recent = higher weight).
      2. Actual kWh so far + projected_daily × days_remaining.

    Args:
        daily_kwh_list    : e.g. [4.2, 4.8, 5.1]  (3 days available)
        days_elapsed      : Integer days elapsed this month.
        rate_per_kwh      : From settings.budget.ratePerKwh
        monthly_budget_kwh: From settings.budget.monthly

    Returns:
        {
          "days_elapsed":          int,
          "days_remaining":        int,
          "actual_kwh_so_far":     float,
          "projected_daily_kwh":   float,
          "projected_remaining_kwh": float,
          "predicted_month_units": float,
          "predicted_cost":        dict,   # full KSEB bill breakdown
          "budget_status":         dict,
          "trend":                 str,    # "increasing"|"stable"|"decreasing"
          "data_points":           int,    # how many days used for forecast
        }
    """
    n = len(daily_kwh_list)
    if n == 0:
        return {"error": "No daily data available for forecast"}

    days_remaining = max(0, 30 - days_elapsed)

    # EWMA: weights = 1, 2, 3 ... n (most recent = highest)
    weights     = list(range(1, n + 1))
    projected   = float(np.average(daily_kwh_list, weights=weights))
    actual_kwh  = sum(daily_kwh_list)

    predicted_remaining = projected * days_remaining
    predicted_total     = actual_kwh + predicted_remaining

    # Trend detection
    if n >= 2:
        slope = np.polyfit(range(n), daily_kwh_list, 1)[0]
        trend = "increasing" if slope > 0.1 else ("decreasing" if slope < -0.1 else "stable")
    else:
        trend = "stable"

    # Cost
    cost = kseb_bill(predicted_total, rate_per_kwh)
    over = predicted_total > monthly_budget_kwh

    return {
        "days_elapsed":              days_elapsed,
        "days_remaining":            days_remaining,
        "actual_kwh_so_far":         round(actual_kwh, 4),
        "projected_daily_kwh":       round(projected, 4),
        "projected_remaining_kwh":   round(predicted_remaining, 4),
        "predicted_month_units":     round(predicted_total, 2),
        "predicted_cost":            cost,
        "budget_status": {
            "monthly_budget_kwh":    monthly_budget_kwh,
            "headroom_kwh":          round(monthly_budget_kwh - predicted_total, 2),
            "over_budget":           over,
        },
        "trend":       trend,
        "data_points": n,
    }


# ── KSEB 2025-2027 tariff ─────────────────────────────────────────────────────

def kseb_bill(monthly_units: float, rate_per_kwh: float = 6.5) -> dict:
    """
    Kerala State Electricity Board 2025-2027 domestic tariff.
    Telescopic slabs up to 250 units; non-telescopic above.

    Returns full bill breakdown.
    """
    units = float(monthly_units)
    energy_charge = _kseb_energy_charge(units)
    fixed_charge  = _kseb_fixed_charge(units)
    duty          = energy_charge * 0.10       # 10% electricity duty

    flat_cost = round(units * rate_per_kwh, 2)
    total     = round(energy_charge + fixed_charge + duty, 2)

    return {
        "monthly_units":        round(units, 2),
        "energy_charge":        round(energy_charge, 2),
        "fixed_charge":         round(fixed_charge, 2),
        "electricity_duty":     round(duty, 2),
        "total_bill_inr":       total,
        "flat_rate_estimate":   flat_cost,    # using settings.budget.ratePerKwh
        "currency":             "INR",
    }


def _kseb_energy_charge(units: float) -> float:
    if units <= 250:
        cost = 0.0
        if units > 200: cost += (units - 200) * 8.50; units = 200
        if units > 150: cost += (units - 150) * 7.20; units = 150
        if units > 100: cost += (units - 100) * 5.35; units = 100
        if units > 50:  cost += (units - 50)  * 4.25; units = 50
        cost += units * 3.35
        return cost
    if units <= 300: return units * 6.75
    if units <= 350: return units * 7.60
    if units <= 400: return units * 7.95
    if units <= 500: return units * 8.25
    return units * 9.20


def _kseb_fixed_charge(units: float) -> float:
    slabs = [(50,50),(100,85),(150,105),(200,140),(250,160),
             (300,220),(350,240),(400,260),(500,285)]
    for limit, charge in slabs:
        if units <= limit:
            return charge
    return 310


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_utc(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    return datetime.now(timezone.utc)


def _ts_span_hours(readings: list[dict]) -> float:
    if len(readings) < 2:
        return 0.0
    t0 = _to_utc(readings[0]["timestamp"])
    t1 = _to_utc(readings[-1]["timestamp"])
    return max(0.0, (t1 - t0).total_seconds() / 3600.0)


def _empty_energy() -> dict:
    return {
        "kwh": 0.0, "avg_power_w": 0.0, "peak_power_w": 0.0,
        "duration_hours": 0.0, "reading_count": 0, "method": "none",
    }
