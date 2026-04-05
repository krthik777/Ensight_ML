"""
api/mongo_client.py
--------------------
MongoDB client that initialises INSTANTLY on import.
- MongoClient() itself is non-blocking in pymongo — it does NOT open a
  TCP connection when you call MongoClient(); it only connects when the
  first DB operation runs.
- A background thread pings Atlas so the startup log is accurate, but
  it never blocks the HTTP server from starting.
- get_db() is O(1) — it just returns the cached reference.
"""

import os
import threading
import yaml
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# ── Module-level singletons (set at import time) ──────────────────────────────
_client: MongoClient | None = None
_db = None
_connected: bool = False  # updated by background ping


def _load_config() -> dict:
    try:
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _build_uri() -> tuple[str, str]:
    """Return (uri, db_name) — env vars override config.yaml."""
    cfg = _load_config().get("mongodb", {})
    uri = os.environ.get(
        "MONGO_URI",
        cfg.get("uri", "mongodb+srv://EnSight:EnSight123@cluster0.vux9jvj.mongodb.net")
    )
    db_name = os.environ.get(
        "MONGO_DB",
        cfg.get("database", "energydashboard")
    )
    return uri, db_name


def _background_ping(uri: str, db_name: str) -> None:
    """Runs in a daemon thread — verifies the Atlas connection and logs result."""
    global _connected
    try:
        _client.admin.command("ping")   # type: ignore[union-attr]
        _connected = True
        print(f"✅ MongoDB connected  →  {db_name}  ({uri[:40]}...)")
    except ServerSelectionTimeoutError as e:
        print(f"⚠️  MongoDB ping timeout: {e}")
        print("   Queries will still be attempted per-request.")
    except Exception as e:
        print(f"⚠️  MongoDB ping failed: {e}")


# ── Initialise client at import time (non-blocking) ───────────────────────────
try:
    _uri, _db_name = _build_uri()

    # MongoClient() is SYNCHRONOUS but only builds internal state — no
    # network I/O happens here.  serverSelectionTimeoutMS controls how long
    # individual DB operations wait if the server is unreachable.
    _client = MongoClient(
        _uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
    )
    _db = _client[_db_name]

    # Ping in the background so the Flask server starts without waiting
    threading.Thread(
        target=_background_ping,
        args=(_uri, _db_name),
        daemon=True,
    ).start()

except Exception as e:
    print(f"❌ MongoDB client creation failed: {e}")
    _client = None
    _db = None


# ── Public API ────────────────────────────────────────────────────────────────

def get_db():
    """
    Returns the MongoDB database instance.
    O(1) — just a reference return.  Never blocks.
    """
    return _db


def is_connected() -> bool:
    """
    Non-blocking connection check using the cached flag set by the
    background ping.  Does NOT issue a new network call.
    """
    return _connected
