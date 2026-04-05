"""
api/app.py
-----------
Flask application factory.
All blueprints are registered here.
"""

from flask import Flask, jsonify, request, g
from datetime import datetime
import json, time

try:
    from api.nilm_routes   import nilm_bp
    from api.energy_routes import energy_bp
    from api.mongo_client  import is_connected
    from api.mongo_data_service import debug_room
except ImportError:
    from .nilm_routes   import nilm_bp
    from .energy_routes import energy_bp
    from .mongo_client  import is_connected
    from .mongo_data_service import debug_room

app = Flask(__name__)

# ── Blueprints ────────────────────────────────────────────────────────────────
app.register_blueprint(nilm_bp)    # /detect-appliances, /predict-appliance-power, /detect-anomaly
app.register_blueprint(energy_bp)  # /energy/today, /energy/daily, /predict-cost, /predict-month-end


# ── Request logger ────────────────────────────────────────────────────────────
@app.before_request
def log_request():
    g.t0 = time.time()
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if request.method == "GET":
        print(f"\n📥 [{ts}] GET {request.path}")
        return

    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    # Summarise long arrays
    display = {}
    for k, v in body.items():
        if isinstance(v, list) and len(v) > 6:
            display[k] = f"[{v[0]}, {v[1]}, {v[2]}, ... +{len(v)-3} more]"
        else:
            display[k] = v

    print(f"\n{'─'*60}")
    print(f"📥 [{ts}]  POST  {request.path}")
    print(f"   Body:")
    for k, v in display.items():
        print(f"     {k}: {v}")
    print(f"{'─'*60}")


@app.after_request
def log_response(response):
    ms   = round((time.time() - g.get("t0", time.time())) * 1000, 1)
    icon = "✅" if response.status_code < 400 else "❌"
    print(f"{icon} [{response.status_code}]  {request.method} {request.path}  ({ms} ms)")
    return response


# ── Core routes ───────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "service":           "EnSight AI Model Engine",
        "status":            "online",
        "mongodb_connected": is_connected(),
        "endpoints": {
            "nilm": [
                "POST /detect-appliances",
                "POST /predict-appliance-power",
                "POST /detect-anomaly",
            ],
            "energy": [
                "POST /energy/today",
                "POST /energy/daily",
                "POST /predict-cost",
                "POST /predict-month-end",
            ],
            "diagnostics": [
                "GET  /health",
                "GET  /db-status",
                "GET  /debug-room?userId=...&roomId=...",
            ],
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":            "healthy",
        "service":           "EnSight AI Model Engine",
        "mongodb_connected": is_connected(),
    })


@app.route("/db-status", methods=["GET"])
def db_status():
    connected = is_connected()
    return jsonify({
        "mongodb_connected": connected,
        "status":            "connected" if connected else "disconnected",
    }), 200 if connected else 503


@app.route("/debug-room", methods=["GET"])
def debug_room_endpoint():
    """
    Diagnose why readings aren't being found.
    Usage: GET /debug-room?userId=<id>&roomId=<id>
    """
    user_id = request.args.get("userId", "")
    room_id = request.args.get("roomId", "")

    if not user_id or not room_id:
        return jsonify({"error": "Pass userId and roomId as query params"}), 400

    result = debug_room(user_id, room_id)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)
