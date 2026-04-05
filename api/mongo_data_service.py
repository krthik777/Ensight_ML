"""
api/mongo_data_service.py
--------------------------
All MongoDB queries for the ML backend.

Mongoose collection → Python collection name mapping:
  PowerReading model  → "powerreadings"
  Settings model      → "settings"
  Room model          → "rooms"
  Appliance model     → "appliances"
  ApplianceDetection  → "appliancedetections"

PowerReading schema:
  userId    ObjectId
  roomId    ObjectId
  voltage   Number
  current   Number
  power     Number   ← mains watts
  energy    Number   ← cumulative kWh (default 0)
  timestamp Date

Indexes: { userId:1, roomId:1, timestamp:-1 }  and  { timestamp:-1 }
"""

from datetime import datetime, timezone, timedelta
from bson import ObjectId
from bson.errors import InvalidId
import yaml
from pathlib import Path

from api.mongo_client import get_db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _oid(id_str) -> ObjectId | None:
    try:
        return ObjectId(str(id_str))
    except (InvalidId, TypeError):
        return None


def _cfg_window() -> int:
    try:
        with open(Path(__file__).parent.parent / "config.yaml") as f:
            return yaml.safe_load(f).get("mongodb", {}).get("sequence_window", 50)
    except Exception:
        return 50


_WINDOW = _cfg_window()


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════

def debug_room(user_id: str, room_id: str) -> dict:
    """
    Diagnostic: count available readings, check timestamps, verify IDs.
    Call GET /db-status?userId=...&roomId=... to see this output.
    """
    db = get_db()
    if db is None:
        return {"error": "DB not connected"}

    user_oid = _oid(user_id)
    room_oid = _oid(room_id)

    if not user_oid or not room_oid:
        return {"error": f"Invalid ObjectId — userId={user_id}, roomId={room_id}"}

    try:
        base = {"userId": user_oid, "roomId": room_oid}
        total_docs = db.powerreadings.count_documents(base)

        latest = db.powerreadings.find_one(base, sort=[("timestamp", -1)])
        oldest = db.powerreadings.find_one(base, sort=[("timestamp", 1)])

        # Also check without userId filter in case userId is mismatched
        room_only_count = db.powerreadings.count_documents({"roomId": room_oid})

        result = {
            "userId":          user_id,
            "roomId":          room_id,
            "total_readings":  total_docs,
            "room_only_count": room_only_count,
            "latest_reading":  {
                "power":     latest.get("power") if latest else None,
                "energy":    latest.get("energy") if latest else None,
                "timestamp": latest["timestamp"].isoformat() if latest else None,
            } if latest else None,
            "oldest_reading": {
                "timestamp": oldest["timestamp"].isoformat() if oldest else None,
            } if oldest else None,
        }

        print(f"🔍 [debug_room] userId={user_id[:8]}.. roomId={room_id[:8]}.. "
              f"total={total_docs} room_only={room_only_count}")
        return result

    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# POWER READINGS — sequences
# ═══════════════════════════════════════════════════════════════════════════════

def get_mains_sequence(user_id: str, room_id: str, limit: int = None) -> list[float]:
    """
    Fetch last `limit` mains power values (Watts) for a room.
    Returns chronological list (oldest→newest) ready for LSTM input.
    """
    limit   = limit or _WINDOW
    db      = get_db()
    user_oid = _oid(user_id)
    room_oid = _oid(room_id)

    if db is None or not user_oid or not room_oid:
        return []

    try:
        docs = list(
            db.powerreadings
            .find({"userId": user_oid, "roomId": room_oid},
                  {"power": 1, "_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        if not docs:
            print(f"⚠️  [mongo] No readings for room {room_id[:8]}.. "
                  f"(try GET /debug-room?userId=...&roomId=... for details)")
            return []

        readings = [float(d["power"]) for d in reversed(docs)]
        print(f"📊 [mongo] Sequence: {len(readings)} readings, "
              f"latest={readings[-1]:.1f}W, avg={sum(readings)/len(readings):.1f}W")
        return readings

    except Exception as e:
        print(f"❌ [mongo] get_mains_sequence: {e}")
        return []


def get_readings_with_timestamps(
    user_id: str,
    room_id: str,
    since: datetime = None,
    until: datetime = None,
    limit: int = 5000,
) -> list[dict]:
    """
    Fetch full reading documents (power, energy, timestamp) for a time window.
    Returns chronological list.  Used by energy analytics for daily kWh.
    """
    db       = get_db()
    user_oid = _oid(user_id)
    room_oid = _oid(room_id)

    if db is None or not user_oid or not room_oid:
        return []

    query = {"userId": user_oid, "roomId": room_oid}
    if since or until:
        ts_filter = {}
        if since: ts_filter["$gte"] = since
        if until: ts_filter["$lte"] = until
        query["timestamp"] = ts_filter

    try:
        docs = list(
            db.powerreadings
            .find(query, {"power": 1, "energy": 1, "timestamp": 1, "_id": 0})
            .sort("timestamp", 1)
            .limit(limit)
        )
        return [
            {
                "power":     float(d.get("power", 0)),
                "energy":    float(d.get("energy", 0)),
                "timestamp": d.get("timestamp"),
            }
            for d in docs
        ]
    except Exception as e:
        print(f"❌ [mongo] get_readings_with_timestamps: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# ENERGY — daily / monthly aggregations
# ═══════════════════════════════════════════════════════════════════════════════

def get_today_readings(user_id: str, room_id: str) -> list[dict]:
    """Return all readings from 00:00 to now (local midnight UTC-adjusted)."""
    now         = datetime.now(timezone.utc)
    day_start   = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return get_readings_with_timestamps(user_id, room_id, since=day_start, until=now)


def get_day_readings(user_id: str, room_id: str, date: datetime) -> list[dict]:
    """Return all readings for a specific calendar day (UTC)."""
    day_start = date.replace(hour=0,  minute=0,  second=0,  microsecond=0, tzinfo=timezone.utc)
    day_end   = date.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
    return get_readings_with_timestamps(user_id, room_id, since=day_start, until=day_end)


def get_recent_daily_kwh_series(
    user_id: str,
    room_id: str,
    days: int = 7,
) -> dict[str, float]:
    """
    Returns a {date_str: kwh} map for the last `days` complete calendar days.
    Uses trapezoidal integration via energy_analytics.compute_daily_kwh().
    """
    from services.energy_analytics import compute_daily_kwh

    result  = {}
    now     = datetime.now(timezone.utc)

    for offset in range(days, 0, -1):          # days ago → yesterday
        target_date = now - timedelta(days=offset)
        readings    = get_day_readings(user_id, room_id, target_date)
        if readings:
            day_data        = compute_daily_kwh(readings)
            date_str        = target_date.strftime("%Y-%m-%d")
            result[date_str] = day_data["kwh"]
            print(f"  📅 {date_str}: {day_data['kwh']:.4f} kWh "
                  f"({day_data['reading_count']} readings, {day_data['method']})")

    return result


def get_monthly_energy_kwh(
    user_id: str,
    room_id: str,
) -> tuple[float | None, int | None]:
    """
    Returns (current_units_this_month_kwh, days_elapsed).
    Sums daily kWh for each day of the current month.
    """
    from services.energy_analytics import compute_daily_kwh

    now         = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    days_elapsed = max(1, (now - month_start).days + 1)

    # Fetch entire month at once
    readings = get_readings_with_timestamps(
        user_id, room_id, since=month_start, until=now, limit=100_000
    )
    if not readings:
        return None, None

    monthly_data = compute_daily_kwh(readings)
    return monthly_data["kwh"], days_elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

_SETTINGS_DEFAULTS = {
    "budget": {"monthly": 400, "currency": "INR", "ratePerKwh": 6.5},
    "notifications": {"pushEnabled": True, "emailEnabled": False, "alertsEnabled": True},
    "autoOptimization": {"enabled": True},
}


def get_user_settings(user_id: str) -> dict:
    import copy
    defaults = copy.deepcopy(_SETTINGS_DEFAULTS)
    db       = get_db()
    user_oid = _oid(user_id)

    if db is None or not user_oid:
        return defaults

    try:
        doc = db.settings.find_one({"userId": user_oid})
        if not doc:
            print(f"⚠️  [mongo] No settings for user {user_id[:8]}.., using defaults")
            return defaults
        defaults["budget"].update(doc.get("budget") or {})
        defaults["notifications"].update(doc.get("notifications") or {})
        if doc.get("autoOptimization"):
            defaults["autoOptimization"].update(doc["autoOptimization"])
        return defaults
    except Exception as e:
        print(f"❌ [mongo] get_user_settings: {e}")
        return defaults


# ═══════════════════════════════════════════════════════════════════════════════
# ROOMS
# ═══════════════════════════════════════════════════════════════════════════════

def get_room_threshold(room_id: str) -> float:
    db       = get_db()
    room_oid = _oid(room_id)
    if db is None or not room_oid:
        return 2000.0
    try:
        doc = db.rooms.find_one({"_id": room_oid}, {"threshold": 1})
        if doc and doc.get("threshold") is not None:
            return float(doc["threshold"])
    except Exception as e:
        print(f"❌ [mongo] get_room_threshold: {e}")
    return 2000.0


# ═══════════════════════════════════════════════════════════════════════════════
# APPLIANCES
# ═══════════════════════════════════════════════════════════════════════════════

def get_room_appliances(user_id: str, room_id: str) -> list[dict]:
    db       = get_db()
    user_oid = _oid(user_id)
    room_oid = _oid(room_id)
    if db is None or not user_oid or not room_oid:
        return []
    try:
        cursor = db.appliances.find(
            {"userId": user_oid, "roomId": room_oid, "isActive": True},
            {"name": 1, "type": 1, "powerRating": 1,
             "estimatedWattage": 1, "usageHoursPerDay": 1, "powerSignature": 1}
        )
        return [
            {
                "name":             d.get("name", "Unknown"),
                "type":             d.get("type", "Other"),
                "powerRating":      d.get("powerRating", 0),
                "estimatedWattage": d.get("estimatedWattage") or d.get("powerRating", 0),
                "usageHoursPerDay": d.get("usageHoursPerDay", 0),
                "powerSignature":   d.get("powerSignature", {}),
            }
            for d in cursor
        ]
    except Exception as e:
        print(f"❌ [mongo] get_room_appliances: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE-BACK
# ═══════════════════════════════════════════════════════════════════════════════

def save_detection_result(
    user_id:          str,
    room_id:          str,
    active_appliances: list[str],
    confidence:       dict,
    power_breakdown:  dict,
    total_power_w:    float,
) -> None:
    """Persist NILM detection to appliancedetections collection."""
    db       = get_db()
    user_oid = _oid(user_id)
    room_oid = _oid(room_id)
    if db is None or not user_oid or not room_oid:
        return
    try:
        db.appliancedetections.insert_one({
            "userId":     user_oid,
            "roomId":     room_oid,
            "appliances": [
                {
                    "name":             app,
                    "confidence":       round(confidence.get(app, 0) * 100, 1),
                    "powerConsumption": round(power_breakdown.get(app, 0), 2),
                }
                for app in active_appliances
            ],
            "totalPower": round(float(total_power_w), 2),
            "timestamp":  datetime.now(timezone.utc),
        })
        print(f"✅ [mongo] Saved detection → {len(active_appliances)} appliances, "
              f"{total_power_w:.1f}W")
    except Exception as e:
        print(f"⚠️  [mongo] save_detection_result: {e}")
