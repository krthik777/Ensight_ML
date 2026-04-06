"""
api/mongo_client.py
--------------------
MongoDB client — ZERO hardcoded credentials.

MONGO_URI must be set as an environment variable:
  - Render  → Dashboard → Environment → MONGO_URI (secret)
  - Local   → Create a .env file (never commit it):
                MONGO_URI=mongodb+srv://EnSight:EnSight123@cluster0...mongodb.net/energydashboard?retryWrites=true&w=majority

The full Atlas URI format MUST be:
  mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/<dbName>?retryWrites=true&w=majority

Note: The ping at startup is SYNCHRONOUS (runs before Flask accepts requests).
This ensures "MongoDB connected" log always appears in Render build/startup logs,
and the health check is accurate from the very first request.
"""

import os
import sys
import yaml
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError

# ── Singletons ─────────────────────────────────────────────────────────────────
_client: MongoClient | None = None
_db     = None
_connected: bool = False


def _load_mongo_uri() -> tuple[str, str]:
    """
    Returns (full_uri, db_name).
    MONGO_URI env var is authoritative. config.yaml is fallback for db_name only.
    Raises RuntimeError if MONGO_URI is not set anywhere.
    """
    # ── env var (Render dashboard / .env) ─────────────────────────────────────
    uri = os.environ.get("MONGO_URI", "").strip()

    # ── fallback: config.yaml uri field (should be blank in production) ────────
    if not uri:
        try:
            cfg_path = Path(__file__).parent.parent / "config.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            uri = cfg.get("mongodb", {}).get("uri", "").strip()
        except Exception:
            pass

    if not uri:
        print("❌ MONGO_URI is not set!", file=sys.stderr)
        print("   Set it in Render → Environment, or create a local .env file.", file=sys.stderr)
        print("   Format: mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/<db>?retryWrites=true&w=majority", file=sys.stderr)
        # Don't crash the server — return empty so get_db() returns None gracefully
        return "", "energydashboard"

    # ── db_name: from MONGO_DB env var or config.yaml or URI path ─────────────
    db_name = os.environ.get("MONGO_DB", "").strip()
    if not db_name:
        try:
            cfg_path = Path(__file__).parent.parent / "config.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            db_name = cfg.get("mongodb", {}).get("database", "energydashboard")
        except Exception:
            db_name = "energydashboard"

    # ── Ensure URI includes db name and Atlas options ──────────────────────────
    # Atlas requires: /dbName?retryWrites=true&w=majority
    if "mongodb+srv" in uri and "?" not in uri:
        uri = f"{uri.rstrip('/')}/{db_name}?retryWrites=true&w=majority&appName=Cluster0"
    elif "mongodb+srv" in uri and db_name not in uri:
        # Has options but no db name in path — insert it
        if "mongodb.net/" not in uri:
            uri = uri.replace(".mongodb.net", f".mongodb.net/{db_name}", 1)

    return uri, db_name


# ── Connect at module import (synchronous ping so Render logs show it clearly) ─
try:
    _uri, _db_name = _load_mongo_uri()

    if _uri:
        print(f"🔌 Connecting to MongoDB Atlas → {_db_name}...")

        _client = MongoClient(
            _uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,
            tls=True,
        )
        _db = _client[_db_name]

        # Synchronous ping — blocks for up to 10 s, but ensures the log
        # "MongoDB connected" appears before gunicorn starts serving requests.
        _client.admin.command("ping")
        _connected = True

        # Only print first 40 chars of URI to avoid leaking credentials in logs
        safe_uri = _uri[:40] + "..." if len(_uri) > 40 else _uri
        print(f"✅ MongoDB connected  →  {_db_name}  ({safe_uri})")

    else:
        print("⚠️  MongoDB skipped — MONGO_URI not configured. DB features disabled.")

except ConfigurationError as e:
    print(f"❌ MongoDB configuration error: {e}", file=sys.stderr)
    print("   Check your MONGO_URI format in Render environment variables.", file=sys.stderr)
    _client = None
    _db     = None

except ServerSelectionTimeoutError as e:
    print(f"❌ MongoDB connection timeout: {e}", file=sys.stderr)
    print("   Possible causes:", file=sys.stderr)
    print("   1. MongoDB Atlas IP Whitelist — add 0.0.0.0/0 in Atlas → Network Access", file=sys.stderr)
    print("   2. Wrong MONGO_URI — check credentials and cluster hostname", file=sys.stderr)
    print("   3. DNS resolution failure (rare on Render)", file=sys.stderr)
    _client    = None
    _db        = None
    _connected = False

except Exception as e:
    print(f"❌ MongoDB unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
    _client    = None
    _db        = None
    _connected = False


# ── Public API ─────────────────────────────────────────────────────────────────

def get_db():
    """Returns the MongoDB database reference. None if not connected."""
    return _db


def is_connected() -> bool:
    """
    Live ping — used only by /health and /db-status.
    Returns cached flag if already connected; does live check if not.
    """
    global _connected
    if _connected:
        return True
    if _client is None:
        return False
    try:
        _client.admin.command("ping")
        _connected = True
        return True
    except Exception:
        return False
