"""
api/energy_routes.py
---------------------
All energy-related endpoints — daily usage, cost calculation, monthly forecast.

POST /energy/today        → today's kWh (midnight to now)
POST /energy/daily        → any specific day's kWh
POST /predict-cost        → KSEB bill from today's actual readings
POST /predict-month-end   → month-end forecast from last 2-7 days of data
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone, timedelta

from services.energy_analytics import compute_daily_kwh, forecast_month_end, kseb_bill

from api.mongo_data_service import (
    get_today_readings,
    get_day_readings,
    get_recent_daily_kwh_series,
    get_monthly_energy_kwh,
    get_user_settings,
)

energy_bp = Blueprint("energy", __name__)


# ── POST /energy/today ────────────────────────────────────────────────────────

@energy_bp.route("/energy/today", methods=["POST"])
def energy_today():
    """
    Today's energy consumption from midnight (00:00 UTC) to now.

    Body: { "userId": "<id>", "roomId": "<id>" }

    Data source: powerreadings.power + powerreadings.timestamp
    Method: trapezoidal integration (or energy field delta if populated)
    """
    try:
        data    = request.get_json() or {}
        user_id = data.get("userId", "")
        room_id = data.get("roomId", "")

        if not user_id or not room_id:
            return jsonify({"success": False, "error": "userId and roomId are required"}), 400

        readings = get_today_readings(user_id, room_id)

        if not readings:
            return jsonify({
                "success":        False,
                "error":          "No readings found for today. Ensure the IoT sensor is streaming.",
                "userId":         user_id,
                "roomId":         room_id,
                "date":           datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }), 404

        energy = compute_daily_kwh(readings)
        settings = get_user_settings(user_id)
        rate     = settings["budget"].get("ratePerKwh", 6.5)
        cost     = kseb_bill(energy["kwh"] * 30, rate)  # extrapolate to monthly for cost

        return jsonify({
            "success":        True,
            "date":           datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "kwh":            energy["kwh"],
            "avg_power_w":    energy["avg_power_w"],
            "peak_power_w":   energy["peak_power_w"],
            "duration_hours": energy["duration_hours"],
            "reading_count":  energy["reading_count"],
            "method":         energy["method"],
            "projected_monthly_units": round(energy["kwh"] * 30, 2),
            "projected_monthly_cost":  cost,
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── POST /energy/daily ────────────────────────────────────────────────────────

@energy_bp.route("/energy/daily", methods=["POST"])
def energy_daily():
    """
    Energy for any specific calendar day (defaults to today).

    Body: { "userId": "<id>", "roomId": "<id>", "date": "YYYY-MM-DD" }
    """
    try:
        data    = request.get_json() or {}
        user_id = data.get("userId", "")
        room_id = data.get("roomId", "")
        date_str = data.get("date")

        if not user_id or not room_id:
            return jsonify({"success": False, "error": "userId and roomId are required"}), 400

        if date_str:
            target = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            target = datetime.now(timezone.utc)

        readings = get_day_readings(user_id, room_id, target)

        if not readings:
            return jsonify({
                "success": False,
                "error":   f"No readings found for {target.strftime('%Y-%m-%d')}",
            }), 404

        energy = compute_daily_kwh(readings)

        return jsonify({
            "success":  True,
            "date":     target.strftime("%Y-%m-%d"),
            **energy,
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── POST /predict-cost ────────────────────────────────────────────────────────

@energy_bp.route("/predict-cost", methods=["POST"])
def predict_cost():
    """
    Calculate monthly KSEB electricity cost based on actual today's readings.
    Extrapolates today's kWh to a 30-day month.

    Body: { "userId": "<id>", "roomId": "<id>" }
    """
    try:
        data    = request.get_json() or {}
        user_id = data.get("userId", "")
        room_id = data.get("roomId", "")

        if not user_id or not room_id:
            return jsonify({"success": False, "error": "userId and roomId are required"}), 400

        settings     = get_user_settings(user_id)
        rate         = settings["budget"].get("ratePerKwh", 6.5)
        monthly_budget = settings["budget"].get("monthly", 400)
        currency     = settings["budget"].get("currency", "INR")

        # Get today's actual readings
        readings = get_today_readings(user_id, room_id)

        if not readings:
            return jsonify({
                "success": False,
                "error":   "No readings for today. Cannot compute cost without data.",
            }), 404

        today_energy = compute_daily_kwh(readings)
        daily_kwh    = today_energy["kwh"]
        monthly_proj = daily_kwh * 30
        cost         = kseb_bill(monthly_proj, rate)

        return jsonify({
            "success":              True,
            "today_kwh":            daily_kwh,
            "today_avg_power_w":    today_energy["avg_power_w"],
            "today_peak_power_w":   today_energy["peak_power_w"],
            "reading_count":        today_energy["reading_count"],
            "estimated_monthly_units": round(monthly_proj, 2),
            "estimated_cost":       cost,
            "currency":             currency,
            "rate_per_kwh_used":    rate,
            "monthly_budget_kwh":   monthly_budget,
            "budget_headroom_kwh":  round(monthly_budget - monthly_proj, 2),
            "over_budget":          monthly_proj > monthly_budget,
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── POST /predict-month-end ───────────────────────────────────────────────────

@energy_bp.route("/predict-month-end", methods=["POST"])
def predict_month_end():
    """
    Forecast end-of-month electricity units and cost.
    Uses EWMA trend from last 2-7 days to project remaining days.

    Body: { "userId": "<id>", "roomId": "<id>", "lookback_days": 7 }

    Data source: powerreadings for each of the last N days.
    Method: Exponentially Weighted Moving Average → linear projection.
    """
    try:
        data          = request.get_json() or {}
        user_id       = data.get("userId", "")
        room_id       = data.get("roomId", "")
        lookback_days = int(data.get("lookback_days", 7))   # how many days to base forecast on
        lookback_days = max(2, min(lookback_days, 14))

        if not user_id or not room_id:
            return jsonify({"success": False, "error": "userId and roomId are required"}), 400

        settings     = get_user_settings(user_id)
        rate         = settings["budget"].get("ratePerKwh", 6.5)
        monthly_bgt  = settings["budget"].get("monthly", 400)
        currency     = settings["budget"].get("currency", "INR")

        # Collect daily kWh for last N days
        print(f"📆 [forecast] Collecting {lookback_days} days of data for forecast...")
        daily_map = get_recent_daily_kwh_series(user_id, room_id, days=lookback_days)

        if len(daily_map) < 2:
            # Fall back to single-day extrapolation
            today_readings = get_today_readings(user_id, room_id)
            if today_readings:
                today_kwh = compute_daily_kwh(today_readings)["kwh"]
                now = datetime.now(timezone.utc)
                days_elapsed = max(1, now.day)
                daily_map = {now.strftime("%Y-%m-%d"): today_kwh}
            else:
                return jsonify({
                    "success": False,
                    "error":   "Insufficient data for forecast. Need at least 1 day of readings.",
                }), 404

        # Days elapsed in current month
        now          = datetime.now(timezone.utc)
        days_elapsed = max(1, now.day)

        daily_values = [v for _, v in sorted(daily_map.items())]

        forecast = forecast_month_end(
            daily_kwh_list    = daily_values,
            days_elapsed      = days_elapsed,
            rate_per_kwh      = rate,
            monthly_budget_kwh= monthly_bgt,
        )

        if "error" in forecast:
            return jsonify({"success": False, "error": forecast["error"]}), 400

        return jsonify({
            "success":  True,
            "currency": currency,
            "daily_breakdown": [
                {"date": d, "kwh": round(v, 4)}
                for d, v in sorted(daily_map.items())
            ],
            **forecast,
            "predicted_cost": forecast["predicted_cost"],   # full kseb breakdown
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
