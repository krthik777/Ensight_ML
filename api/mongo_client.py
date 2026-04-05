"""
api/mongo_client.py
--------------------
MongoDB client for production (Render) and local use.

Fix log:
  - URI now appended with /databaseName?retryWrites=true&w=majority (Atlas requirement)
  - is_connected() does a live ping instead of relying on a background-thread flag
    (the flag caused false "not connected" on Render health checks that fired before
     the ping thread completed)
  - Increased serverSelectionTimeoutMS to 8000 for Render's cold-start latency
"""

import os
import threading
import yaml
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# ── Singletons ─────────────────────────────────────────────────────────────────
_client: MongoClient | None = None
_db     = None


def _load_config() -> dict:
    try:
        with open(Path(__file__).parent.parent / "config.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _build_uri() -> tuple[str, str]:
    """
    Return (full_uri, db_name).

    Atlas SRV URIs MUST include:
      /databaseName?retryWrites=true&w=majority
    otherwise some drivers fail to authenticate.

    If the user's MONGO_URI already contains '?' we don't double-append.
    """
    cfg     = _load_config().get("mongodb", {})
    base_uri = os.environ.get(
        "MONGO_URI",
        cfg.get("uri", "mongodb+srv://EnSight:EnSight123@cluster0.vux9jvj.mongodb.net")
    )
    db_name = os.environ.get(
        "MONGO_DB",
        cfg.get("database", "energydashboard")
    )

    # Build the full URI with db name + Atlas-required options if not already set
    if "?" not in base_uri and "mongodb+srv" in base_uri:
        # Strip trailing slash then append
        full_uri = f"{base_uri.rstrip('/')}/{db_name}?retryWrites=true&w=majority&appName=Cluster0"
    else:
        full_uri = base_uri

    return full_uri, db_name


def _warm_up_ping(db_name: str) -> None:
    """Background thread — logs connection result at startup."""
    try:
        _client.admin.command("ping")   # type: ignore[union-attr]
        print(f"✅ MongoDB connected  →  {db_name}")
    except Exception as e:
        print(f"⚠️  MongoDB startup ping failed (will retry on first request): {e}")


# ── Initialise at import time (non-blocking) ──────────────────────────────────
try:
    _uri, _db_name = _build_uri()

    _client = MongoClient(
        _uri,
        serverSelectionTimeoutMS=8000,   # Render cold-start can be slow
        connectTimeoutMS=8000,
        socketTimeoutMS=15000,
        tls=True,                        # Atlas always requires TLS
    )
    _db = _client[_db_name]

    # Ping in background — never blocks gunicorn worker startup
    threading.Thread(
        target=_warm_up_ping,
        args=(_db_name,),
        daemon=True,
    ).start()

except Exception as e:
    print(f"❌ MongoDB client init failed: {e}")
    _client = None
    _db     = None


# ── Public API ─────────────────────────────────────────────────────────────────

def get_db():
    """Returns the MongoDB database reference (O(1), no network call)."""
    return _db


def is_connected() -> bool:
    """
    Live connection check — issues a lightweight ping to Atlas.
    Used only by /health and /db-status endpoints (not per-request).
    Returns False gracefully if DB is unreachable.
    """
    if _client is None:
        return False
    try:
        _client.admin.command("ping")
        return True
    except Exception:
        return False
