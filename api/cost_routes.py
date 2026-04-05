"""
api/cost_routes.py
-------------------
Cost prediction endpoint.

Minimal body — only userId + roomId required.
Daily usage is computed from MongoDB powerreadings.energy delta (last 24 h).
The KSEB rate is overridden by the user's settings.budget.ratePerKwh when
it differs from the default slab calculation.

Endpoint
--------
POST /predict-cost  →  estimated_monthly_units + KSEB bill in INR
"""

from flask import Blueprint, request, jsonify
from services.cost_prediction import cost_prediction_service
from api.mongo_data_service import get_daily_energy_kwh, get_user_settings

cost_bp = Blueprint("cost", __name__)


@cost_bp.route("/predict-cost", methods=["POST"])
def predict_cost():
    """
    Predict monthly electricity cost.

    Minimal body:
        { "userId": "<id>", "roomId": "<id>" }

    Optional override (for testing without DB):
        { "userId": "...", "roomId": "...", "daily_usage_kwh": 8.2 }

    Data sourced from MongoDB:
        • daily_usage_kwh  ← powerreadings.energy delta over 24 h
        • rate_per_kwh     ← settings.budget.ratePerKwh
        • budget_monthly   ← settings.budget.monthly (for headroom calc)
    """
    try:
        data = request.get_json() or {}

        user_id = data.get("userId", "")
        room_id = data.get("roomId", "")

        # ── Resolve daily usage ──────────────────────────────────────────
        daily_usage_kwh = data.get("daily_usage_kwh")  # optional direct override

        if daily_usage_kwh is None:
            if not user_id or not room_id:
                return jsonify({
                    "success": False,
                    "error": "Provide 'userId' + 'roomId', or 'daily_usage_kwh' directly",
                }), 400

            daily_usage_kwh = get_daily_energy_kwh(user_id, room_id)

            if daily_usage_kwh is None:
                return jsonify({
                    "success": False,
                    "error": "No power readings found for this room in the last 24 hours. "
                             "Ensure the sensor is streaming data.",
                }), 404

            print(f"📊 [cost_routes] Daily kWh from MongoDB: {daily_usage_kwh}")

        # ── Resolve user settings ────────────────────────────────────────
        settings = get_user_settings(user_id) if user_id else {}
        rate_per_kwh = settings.get("budget", {}).get("ratePerKwh", 6.5)
        monthly_budget_kwh = settings.get("budget", {}).get("monthly", 400)
        currency = settings.get("budget", {}).get("currency", "INR")

        # ── Calculate cost ──────────────────────────────────────────────
        result = cost_prediction_service.calculate_monthly_cost(
            daily_usage_kwh, rate_per_kwh=rate_per_kwh
        )

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]}), 400

        # Augment with budget headroom info
        projected_units = result["estimated_monthly_units"]
        result["success"] = True
        result["currency"] = currency
        result["rate_per_kwh_used"] = rate_per_kwh
        result["monthly_budget_kwh"] = monthly_budget_kwh
        result["budget_headroom_kwh"] = round(monthly_budget_kwh - projected_units, 2)
        result["over_budget"] = projected_units > monthly_budget_kwh

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
