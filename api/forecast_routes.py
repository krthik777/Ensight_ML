"""
api/forecast_routes.py
-----------------------
Month-end forecasting endpoint.

Minimal body — only userId + roomId required.
current_units and days_elapsed are calculated from MongoDB powerreadings.

Endpoint
--------
POST /predict-month-end  →  predicted month-end units + expected cost
"""

from flask import Blueprint, request, jsonify
from services.month_forecast import month_forecast_service
from api.mongo_data_service import get_monthly_energy_kwh, get_user_settings

forecast_bp = Blueprint("forecast", __name__)


@forecast_bp.route("/predict-month-end", methods=["POST"])
def predict_month_end():
    """
    Forecast end-of-month electricity consumption and cost.

    Minimal body:
        { "userId": "<id>", "roomId": "<id>" }

    Optional overrides (for testing):
        { "userId": "...", "roomId": "...",
          "current_units": 120, "days_elapsed": 15 }

    Data sourced from MongoDB:
        • current_units  ← powerreadings.energy delta since 1st of month
        • days_elapsed   ← calculated from current date automatically
        • ratePerKwh     ← settings.budget.ratePerKwh
        • monthly_budget ← settings.budget.monthly
    """
    try:
        data = request.get_json() or {}

        user_id = data.get("userId", "")
        room_id = data.get("roomId", "")

        # ── Resolve current_units & days_elapsed ─────────────────────────
        current_units = data.get("current_units")
        days_elapsed = data.get("days_elapsed")

        if current_units is None or days_elapsed is None:
            if not user_id or not room_id:
                return jsonify({
                    "success": False,
                    "error": "Provide 'userId' + 'roomId', or supply "
                             "'current_units' and 'days_elapsed' directly",
                }), 400

            db_units, db_days = get_monthly_energy_kwh(user_id, room_id)

            if db_units is None or db_days is None:
                return jsonify({
                    "success": False,
                    "error": "No monthly power data found for this room. "
                             "Ensure the sensor has been streaming data since the 1st of the month.",
                }), 404

            current_units = db_units if current_units is None else current_units
            days_elapsed = db_days if days_elapsed is None else days_elapsed
            print(
                f"📊 [forecast_routes] MongoDB → {current_units} kWh over {days_elapsed} days"
            )

        # ── Resolve user settings ────────────────────────────────────────
        settings = get_user_settings(user_id) if user_id else {}
        rate_per_kwh = settings.get("budget", {}).get("ratePerKwh", 6.5)
        monthly_budget_kwh = settings.get("budget", {}).get("monthly", 400)
        currency = settings.get("budget", {}).get("currency", "INR")

        # ── Forecast ─────────────────────────────────────────────────────
        result = month_forecast_service.predict_month_end(
            current_units, days_elapsed, rate_per_kwh=rate_per_kwh
        )

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]}), 400

        # Augment with budget comparison
        predicted_units = result["predicted_month_units"]
        result["success"] = True
        result["currency"] = currency
        result["rate_per_kwh_used"] = rate_per_kwh
        result["monthly_budget_kwh"] = monthly_budget_kwh
        result["budget_headroom_kwh"] = round(monthly_budget_kwh - predicted_units, 2)
        result["over_budget"] = predicted_units > monthly_budget_kwh
        result["days_elapsed"] = days_elapsed
        result["days_remaining"] = max(0, 30 - days_elapsed)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
