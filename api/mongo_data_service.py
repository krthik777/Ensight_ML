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

Bug-fix log:
  - _oid() now returns None for empty/None strings (prevents ObjectId("") crash)
  - get_room_appliances: removed isActive filter — many appliances don't set this flag
  - get_today_readings: uses IST midnight (UTC+5:30) instead of UTC midnight
    so "today" matches the user's actual day in India
  - get_recent_daily_kwh_series: iterates yesterday → (days ago), includes today
  - get_user_settings: budget.monthly is now treated as kWh budget not INR
"""

from datetime import datetime, timezone, timedelta
from bson import ObjectId
from bson.errors import InvalidId
import yaml
from pathlib import Path

from api.mongo_client import get_db


# ── Helpers ───────────────────────────────────────────────────────────────────

IST_OFFSET = timedelta(hours=5, minutes=30)   # India Standard Time = UTC+5:30


def _oid(id_str) -> "ObjectId | None":
    """Convert string to ObjectId. Returns None for empty/invalid strings."""
    if not id_str:
        return None
    try:
        return ObjectId(str(id_str))
    except (InvalidId, TypeError):
        return None


def _cfg_window() -> int:
    try:
        with open(Path(__file__).parent.parent / "config.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f).get("mongodb", {}).get("sequence_window", 50)
    except Exception as e:
        print(f"⚠️ Error loading config: {e}. Using defaults.")
        return 50


_WINDOW = _cfg_window()

# Possible Mongoose collection names for PowerReading model
# Mongoose default: lowercase + plural = 'powerreadings'
# Some setups use custom names — we probe all candidates
_READING_COLLECTION_CANDIDATES = [
    "powerreadings",
    "PowerReadings",
    "power_readings",
    "powerReading",
    "power_reading",
    "readings",
]
_readings_collection_name: str = "powerreadings"   # updated by _detect_collection()


def _detect_collection(db) -> str:
    """
    Auto-detect the correct collection name for power readings.
    Lists all collections in the DB, finds which one has documents
    that look like power readings (has 'power' and 'voltage' fields).

    Result is cached in _readings_collection_name.
    """
    global _readings_collection_name
    try:
        actual_names = db.list_collection_names()
        print(f"📋 [mongo] Collections in DB: {actual_names}")

        # First: try exact known candidates
        for candidate in _READING_COLLECTION_CANDIDATES:
            if candidate in actual_names:
                # Verify it actually has power-reading documents
                sample = db[candidate].find_one({"power": {"$exists": True}})
                if sample:
                    _readings_collection_name = candidate
                    print(f"✅ [mongo] Power readings collection detected: '{candidate}'")
                    return candidate

        # Second: scan all collections for any that have a 'power' field
        for name in actual_names:
            sample = db[name].find_one({"power": {"$exists": True}, "voltage": {"$exists": True}})
            if sample:
                _readings_collection_name = name
                print(f"✅ [mongo] Power readings found in collection: '{name}' (auto-detected)")
                return name

        print(f"⚠️  [mongo] Could not find power readings in any collection.")
        print(f"   Available collections: {actual_names}")
        return "powerreadings"   # default fallback

    except Exception as e:
        print(f"❌ [mongo] _detect_collection error: {e}")
        return "powerreadings"


def _get_readings_col(db):
    """
    Returns the correct pymongo collection object for power readings.
    Detects the name on first call, then uses cached value.
    """
    global _readings_collection_name
    col = db[_readings_collection_name]
    # Quick check: if the cached collection appears empty, re-detect
    try:
        if col.estimated_document_count() == 0:
            _detect_collection(db)
            col = db[_readings_collection_name]
    except Exception:
        pass
    return col


def _ist_day_bounds_utc(date_utc: datetime) -> "tuple[datetime, datetime]":
    """
    Given a UTC datetime, return the UTC start/end of the IST calendar day.
    IST midnight = UTC 18:30 of the previous day.
    """
    ist_dt  = date_utc + IST_OFFSET
    ist_day = ist_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    utc_start = ist_day - IST_OFFSET
    utc_end   = utc_start + timedelta(days=1) - timedelta(microseconds=1)
    return utc_start.replace(tzinfo=timezone.utc), utc_end.replace(tzinfo=timezone.utc)


def _get_actual_room_ids_for_user(db, user_oid) -> list:
    """
    Returns list of distinct roomIds that have powerreadings for this user.
    Used to diagnose roomId mismatches — shown in logs when a query returns empty.
    """
    try:
        col = _get_readings_col(db)
        ids = col.distinct("roomId", {"userId": user_oid})
        return [str(rid) for rid in ids]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC — debug_room
# ═══════════════════════════════════════════════════════════════════════════════

def debug_room(user_id: str, room_id: str) -> dict:
    """
    Diagnostic: count available readings and list all roomIds with data for
    this user — makes roomId mismatches immediately visible.
    Call GET /debug-room?userId=...&roomId=... 
    """
    db = get_db()
    if db is None:
        return {"error": "DB not connected"}

    user_oid = _oid(user_id)
    room_oid = _oid(room_id)

    if not user_oid or not room_oid:
        return {"error": f"Invalid ObjectId — userId={user_id}, roomId={room_id}"}

    try:
        col        = _get_readings_col(db)
        base       = {"userId": user_oid, "roomId": room_oid}
        total_docs = col.count_documents(base)

        latest = col.find_one(base, sort=[("timestamp", -1)])
        oldest = col.find_one(base, sort=[("timestamp", 1)])

        room_only_count = col.count_documents({"roomId": room_oid})
        appliance_count = db.appliances.count_documents(
            {"userId": user_oid, "roomId": room_oid}
        )

        all_room_ids = _get_actual_room_ids_for_user(db, user_oid)

        now = datetime.now(timezone.utc)
        day_start_utc, day_end_utc = _ist_day_bounds_utc(now)
        today_count = col.count_documents({
            **base,
            "timestamp": {"$gte": day_start_utc, "$lte": day_end_utc},
        })

        result = {
            "userId":           user_id,
            "roomId_queried":   room_id,
            "total_readings":   total_docs,
            "today_readings":   today_count,
            "room_only_count":  room_only_count,
            "appliance_count":  appliance_count,
            "roomIds_with_data_for_user": all_room_ids,   # ← KEY: shows which roomIds have readings
            "ist_day_window": {
                "start_utc": day_start_utc.isoformat(),
                "end_utc":   day_end_utc.isoformat(),
            },
            "latest_reading": {
                "power":     latest.get("power") if latest else None,
                "energy":    latest.get("energy") if latest else None,
                "timestamp": latest["timestamp"].isoformat() if latest else None,
            } if latest else None,
            "oldest_reading": {
                "timestamp": oldest["timestamp"].isoformat() if oldest else None,
            } if oldest else None,
        }

        if total_docs == 0 and all_room_ids:
            result["mismatch_warning"] = (
                f"No readings for roomId={room_id}, but this user has readings in "
                f"{len(all_room_ids)} other room(s): {all_room_ids}. "
                f"Check that Node.js is sending the correct roomId."
            )

        print(f"🔍 [debug_room] userId={user_id[:8]}.. roomId={room_id[:8]}.. "
              f"total={total_docs} today={today_count} appliances={appliance_count}")
        if all_room_ids:
            print(f"   Rooms with data for this user: {all_room_ids}")
        return result

    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# POWER READINGS — sequences
# ═══════════════════════════════════════════════════════════════════════════════

def get_mains_sequence(user_id: str, room_id: str, limit: int = None) -> list:
    """
    Fetch last `limit` mains power values (Watts) for a room.
    Returns chronological list (oldest→newest) ready for LSTM input.

    If no readings are found, logs the actual roomIds that DO have data
    for this user so the mismatch is immediately visible in terminal logs.
    """
    limit    = limit or _WINDOW
    db       = get_db()
    user_oid = _oid(user_id)
    room_oid = _oid(room_id)

    if db is None or not user_oid or not room_oid:
        return []

    try:
        col = _get_readings_col(db)
        docs = list(
            col
            .find({"userId": user_oid, "roomId": room_oid},
                  {"power": 1, "_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        if not docs:
            actual_ids = _get_actual_room_ids_for_user(db, user_oid)
            print(f"\u26a0\ufe0f  [mongo] No readings — collection='{_readings_collection_name}' userId={user_id[:8]}.. roomId={room_id[:8]}..")
            if actual_ids:
                print(f"   \u2757 RoomIds WITH data: {actual_ids}")
                if room_id not in actual_ids:
                    print(f"   \u2757 roomId mismatch! Sent={room_id}, available={actual_ids}")
            else:
                print(f"   No readings for this user in any room. Is the IoT device streaming?")
            return []

        readings = [float(d.get("power", 0)) for d in reversed(docs)]
        print(f"\U0001f4ca [mongo] Sequence: {len(readings)} readings from '{_readings_collection_name}', "
              f"latest={readings[-1]:.1f}W, avg={sum(readings)/len(readings):.1f}W")
        return readings

    except Exception as e:
        print(f"\u274c [mongo] get_mains_sequence: {e}")
        return []


def get_readings_with_timestamps(
    user_id: str,
    room_id: str,
    since: datetime = None,
    until: datetime = None,
    limit: int = 5000,
) -> list:
    """
    Fetch full reading documents (power, energy, timestamp) for a time window.
    Returns chronological list sorted oldest→newest.
    Used by energy analytics for daily kWh calculation.
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
        col  = _get_readings_col(db)
        docs = list(
            col
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
        print(f"\u274c [mongo] get_readings_with_timestamps: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# ENERGY — daily / monthly aggregations
# ═══════════════════════════════════════════════════════════════════════════════

def get_today_readings(user_id: str, room_id: str) -> list:
    """
    Return all readings for today in IST (India Standard Time).

    Bug-fix: UTC midnight was wrong for Indian users — a reading at 23:00 IST
    on 4 May = 17:30 UTC on 4 May, which falls BEFORE UTC midnight. It would
    be missed if we used UTC 00:00 as the day boundary.

    We now use IST midnight converted to UTC:
      IST 00:00 = UTC 18:30 of the previous day.
    """
    now = datetime.now(timezone.utc)
    day_start_utc, day_end_utc = _ist_day_bounds_utc(now)
    print(f"📅 [energy] Today IST window: "
          f"{(day_start_utc + IST_OFFSET).strftime('%Y-%m-%d %H:%M')} IST → "
          f"{(day_end_utc + IST_OFFSET).strftime('%Y-%m-%d %H:%M')} IST "
          f"(UTC: {day_start_utc.strftime('%H:%M')} → {day_end_utc.strftime('%H:%M')})")
    return get_readings_with_timestamps(
        user_id, room_id, since=day_start_utc, until=day_end_utc
    )


def get_day_readings(user_id: str, room_id: str, date: datetime) -> list:
    """
    Return all readings for a specific calendar day (IST-aware).
    `date` can be any timezone; we find the IST day it represents.
    """
    # Ensure it's in UTC first
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    # Find the IST day bounds for this date
    day_start_utc, day_end_utc = _ist_day_bounds_utc(date)
    return get_readings_with_timestamps(
        user_id, room_id, since=day_start_utc, until=day_end_utc
    )


def get_recent_daily_kwh_series(
    user_id: str,
    room_id: str,
    days: int = 7,
) -> dict:
    """
    Returns a {date_str: kwh} map for the last `days` calendar days (IST).
    Includes today and the N-1 days before it.

    Bug-fix: Previously iterated 'days ago → 1 day ago' (excluded today).
    Now includes today's partial data, which is critical for forecasting.
    """
    from services.energy_analytics import compute_daily_kwh

    result = {}
    now    = datetime.now(timezone.utc)
    ist_now = now + IST_OFFSET

    for offset in range(days - 1, -1, -1):          # days-1 days ago → today
        target_ist  = ist_now - timedelta(days=offset)
        target_utc  = target_ist - IST_OFFSET        # back to UTC for DB query
        target_utc  = target_utc.replace(tzinfo=timezone.utc)

        readings = get_day_readings(user_id, room_id, target_utc)
        if readings:
            day_data = compute_daily_kwh(readings)
            date_str = target_ist.strftime("%Y-%m-%d")
            result[date_str] = day_data["kwh"]
            print(f"  📅 {date_str}: {day_data['kwh']:.4f} kWh "
                  f"({day_data['reading_count']} readings, {day_data['method']})")
        else:
            date_str = target_ist.strftime("%Y-%m-%d")
            print(f"  📅 {date_str}: no readings")

    return result


def get_monthly_energy_kwh(
    user_id: str,
    room_id: str,
) -> "tuple[float | None, int | None]":
    """
    Returns (current_units_this_month_kwh, days_elapsed).
    Fetches from IST month start.
    """
    from services.energy_analytics import compute_daily_kwh

    now     = datetime.now(timezone.utc)
    ist_now = now + IST_OFFSET
    # IST month start → UTC
    ist_month_start = ist_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    utc_month_start = (ist_month_start - IST_OFFSET).replace(tzinfo=timezone.utc)
    days_elapsed    = max(1, ist_now.day)

    readings = get_readings_with_timestamps(
        user_id, room_id, since=utc_month_start, until=now, limit=100_000
    )
    if not readings:
        return None, None

    monthly_data = compute_daily_kwh(readings)
    return monthly_data["kwh"], days_elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

_SETTINGS_DEFAULTS = {
    "budget": {
        "monthly":    400,     # monthly kWh budget
        "currency":   "INR",
        "ratePerKwh": 6.5,     # INR per kWh (used as fallback; KSEB tariff takes precedence)
    },
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
        if doc.get("budget"):
            defaults["budget"].update(doc["budget"])
        if doc.get("notifications"):
            defaults["notifications"].update(doc["notifications"])
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

def get_room_appliances(user_id: str, room_id: str) -> list:
    """
    Fetch active appliances for a room.

    Bug-fix: Removed `isActive: True` filter — many appliances in MongoDB
    don't have this field set (defaults to undefined/null in Mongoose),
    causing the query to return [] even when appliances exist.
    We now fetch all appliances and filter client-side, treating missing
    isActive as True (default active).
    """
    db       = get_db()
    user_oid = _oid(user_id)
    room_oid = _oid(room_id)
    if db is None or not user_oid or not room_oid:
        return []
    try:
        # Fetch all appliances for this room (not filtered by isActive)
        cursor = db.appliances.find(
            {"userId": user_oid, "roomId": room_oid},
            {"name": 1, "type": 1, "powerRating": 1,
             "estimatedWattage": 1, "usageHoursPerDay": 1, "powerSignature": 1,
             "isActive": 1}
        )
        results = []
        for d in cursor:
            # Treat missing isActive as True (default), only skip explicit False
            if d.get("isActive") is False:
                continue
            results.append({
                "name":             d.get("name", "Unknown"),
                "type":             d.get("type", "Other"),
                "powerRating":      float(d.get("powerRating") or 0),
                "estimatedWattage": float(d.get("estimatedWattage") or d.get("powerRating") or 0),
                "usageHoursPerDay": float(d.get("usageHoursPerDay") or 0),
                "powerSignature":   d.get("powerSignature") or {},
            })

        print(f"🔌 [mongo] get_room_appliances: {len(results)} appliances found "
              f"for room {room_id[:8]}..")
        return results
    except Exception as e:
        print(f"❌ [mongo] get_room_appliances: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE-BACK
# ═══════════════════════════════════════════════════════════════════════════════

def save_detection_result(
    user_id:           str,
    room_id:           str,
    active_appliances: list,
    confidence:        dict,
    power_breakdown:   dict,
    total_power_w:     float,
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
                    "confidence":       round(float(confidence.get(app, 0)) * 100, 1),
                    "powerConsumption": round(float(power_breakdown.get(app, 0)), 2),
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
