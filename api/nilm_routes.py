"""
api/nilm_routes.py
-------------------
POST /detect-appliances      → smart NILM (Tier 1: DB matching, Tier 2: LSTM, Tier 3: heuristic)
POST /predict-appliance-power → power breakdown per appliance
POST /detect-anomaly          → spike detection using DB threshold
"""

from flask import Blueprint, request, jsonify

from services.nilm_inference import nilm_inference_service
from services.nilm_engine import smart_nilm_detect

from api.mongo_data_service import (
    get_mains_sequence,
    get_room_appliances,
    get_room_threshold,
    save_detection_result,
)

nilm_bp = Blueprint("nilm", __name__)


# ── POST /detect-appliances ───────────────────────────────────────────────────

@nilm_bp.route("/detect-appliances", methods=["POST"])
def detect_appliances():
    """
    Smart NILM appliance detection.

    Body: { "userId": "<id>", "roomId": "<id>" }

    What happens:
      1. Fetches last 50 power readings from powerreadings (MongoDB)
      2. Fetches registered appliances for the room from appliances (MongoDB)
      3. Runs smart_nilm_detect (DB matching → LSTM → heuristic)
      4. Saves result to appliancedetections (MongoDB)
    """
    try:
        data    = request.get_json() or {}
        user_id = data.get("userId", "")
        room_id = data.get("roomId", "")

        if not user_id or not room_id:
            return jsonify({"success": False, "error": "userId and roomId are required"}), 400

        # ── Fetch live sequence from MongoDB ─────────────────────────────────
        mains_sequence = get_mains_sequence(user_id, room_id)

        # ── Fetch registered appliances for smart matching ────────────────────
        db_appliances  = get_room_appliances(user_id, room_id)

        # ── Run smart NILM ────────────────────────────────────────────────────
        result = smart_nilm_detect(mains_sequence, db_appliances, nilm_inference_service)

        # ── Persist to MongoDB ────────────────────────────────────────────────
        if result.get("active_appliances"):
            save_detection_result(
                user_id          = user_id,
                room_id          = room_id,
                active_appliances= result["active_appliances"],
                confidence       = result.get("confidence", {}),
                power_breakdown  = result.get("power_breakdown", {}),
                total_power_w    = result.get("total_power_w", result.get("mean_power_w", 0)),
            )

        return jsonify({
            "success":          True,
            "active_appliances": result["active_appliances"],
            "confidence":        result["confidence"],
            "power_breakdown_w": result["power_breakdown"],
            "total_power_w":     result.get("total_power_w", result.get("mean_power_w", 0)),
            "mean_power_w":      result.get("mean_power_w", 0),
            "unmatched_w":       result.get("unmatched_w", 0),
            "detection_tier":    result.get("tier", "unknown"),
            "db_appliances_count": len(db_appliances),
            "readings_used":     len(mains_sequence),
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── POST /predict-appliance-power ─────────────────────────────────────────────

@nilm_bp.route("/predict-appliance-power", methods=["POST"])
def predict_appliance_power():
    """
    Per-appliance power breakdown using the best available method.
    Same as detect-appliances but returns the power_breakdown as the primary field.

    Body: { "userId": "<id>", "roomId": "<id>" }
    """
    try:
        data    = request.get_json() or {}
        user_id = data.get("userId", "")
        room_id = data.get("roomId", "")

        if not user_id or not room_id:
            return jsonify({"success": False, "error": "userId and roomId are required"}), 400

        mains_sequence = get_mains_sequence(user_id, room_id)
        db_appliances  = get_room_appliances(user_id, room_id)
        result         = smart_nilm_detect(mains_sequence, db_appliances, nilm_inference_service)

        return jsonify({
            "success":       True,
            "appliances":    result["power_breakdown"],
            "total_power_w": result.get("total_power_w", result.get("mean_power_w", 0)),
            "tier":          result.get("tier", "unknown"),
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── POST /detect-anomaly ──────────────────────────────────────────────────────

@nilm_bp.route("/detect-anomaly", methods=["POST"])
def detect_anomaly():
    """
    Power anomaly detection using 3-sigma statistical test AND
    the room's configured threshold from the rooms collection.

    Body: { "userId": "<id>", "roomId": "<id>" }
    """
    try:
        data    = request.get_json() or {}
        user_id = data.get("userId", "")
        room_id = data.get("roomId", "")

        if not user_id or not room_id:
            return jsonify({"success": False, "error": "userId and roomId are required"}), 400

        mains_sequence  = get_mains_sequence(user_id, room_id)
        room_threshold  = get_room_threshold(room_id)
        result          = nilm_inference_service.detect_anomaly(
            mains_sequence, room_threshold=room_threshold
        )

        result["success"]             = True
        result["room_threshold_used"] = room_threshold
        result["readings_used"]       = len(mains_sequence)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
