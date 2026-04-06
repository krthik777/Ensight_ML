"""
api/mongo_client.py
--------------------
MongoDB client — ZERO hardcoded credentials.

MONGO_URI must be set as an environment variable:
  - Render  → Dashboard → Environment → MONGO_URI (secret)
  - Local   → .env file (never commit):
      MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/db?retryWrites=true&w=majority

Key design decisions:
  - Startup ping failure does NOT null out _client — the client stays alive
    so that is_connected() live-pings can succeed on subsequent calls.
  - PYTHONUNBUFFERED=1 must be set in Render env vars; flush=True is also
    used on every print as a belt-and-suspenders safeguard.
"""

import os
import sys
import yaml
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError

# ── Singletons ─────────────────────────────────────────────────────────────────
_client = None   # MongoClient instance — kept alive even after startup ping failure
_db     = None
_connected: bool = False


def _load_mongo_uri():
    """
    Returns (full_uri, db_name).
    MONGO_URI env var is the authoritative source.
    config.yaml is fallback for db_name only (uri field intentionally blank there).
    """
    uri = os.environ.get("MONGO_URI", "").strip()

    if not uri:
        try:
            cfg_path = Path(__file__).parent.parent / "config.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            uri = cfg.get("mongodb", {}).get("uri", "").strip()
        except Exception:
            pass

    db_name = os.environ.get("MONGO_DB", "").strip()
    if not db_name:
        try:
            cfg_path = Path(__file__).parent.parent / "config.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            db_name = cfg.get("mongodb", {}).get("database", "energydashboard")
        except Exception:
            db_name = "energydashboard"

    if not uri:
        print("❌ [MongoDB] MONGO_URI is not set!", flush=True, file=sys.stderr)
        print("   Render → Environment → Add MONGO_URI as a secret", flush=True, file=sys.stderr)
        print("   Format: mongodb+srv://user:pass@cluster.mongodb.net/db?retryWrites=true&w=majority", flush=True, file=sys.stderr)
        return "", db_name

    # Append Atlas options if missing
    if "mongodb+srv" in uri and "?" not in uri:
        uri = f"{uri.rstrip('/')}/{db_name}?retryWrites=true&w=majority&appName=Cluster0"

    return uri, db_name


# ── Connect at module import ──────────────────────────────────────────────────
try:
    _uri, _db_name = _load_mongo_uri()

    if _uri:
        safe_uri = _uri[:50] + "..." if len(_uri) > 50 else _uri
        print(f"🔌 [MongoDB] Connecting → {_db_name}", flush=True)
        print(f"   {safe_uri}", flush=True)

        # Create client — kept alive regardless of ping outcome
        _client = MongoClient(
            _uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,
            tls=True,
        )
        _db = _client[_db_name]

        # Startup ping — sets _connected flag but does NOT null _client on failure.
        # If this ping times out (e.g. two workers racing on startup), the client
        # stays alive and is_connected() will retry on the next request.
        try:
            _client.admin.command("ping")
            _connected = True
            print(f"✅ [MongoDB] Connected → {_db_name}", flush=True)
        except ServerSelectionTimeoutError as ping_err:
            # _client and _db stay alive — is_connected() will retry
            print(f"⚠️  [MongoDB] Startup ping timed out (will retry on requests): {ping_err}", flush=True, file=sys.stderr)
            print("   If this persists → Atlas Network Access → add 0.0.0.0/0", flush=True, file=sys.stderr)
        except Exception as ping_err:
            print(f"⚠️  [MongoDB] Startup ping failed (will retry on requests): {ping_err}", flush=True, file=sys.stderr)

    else:
        print("⚠️  [MongoDB] Skipped — MONGO_URI not set. DB endpoints disabled.", flush=True)

except ConfigurationError as e:
    print(f"❌ [MongoDB] Bad URI format: {e}", flush=True, file=sys.stderr)
    print("   Check MONGO_URI value in Render Environment settings.", flush=True, file=sys.stderr)
    _client = None
    _db     = None

except Exception as e:
    print(f"❌ [MongoDB] Init error — {type(e).__name__}: {e}", flush=True, file=sys.stderr)
    _client = None
    _db     = None


# ── Public API ─────────────────────────────────────────────────────────────────

def get_db():
    """Returns the MongoDB database reference. None if not connected."""
    return _db


def is_connected() -> bool:
    """
    Returns True if MongoDB is reachable.

    Uses cached _connected flag if startup ping succeeded.
    Falls back to a live ping if startup ping timed out (common when two
    workers race on startup without --preload).

    Called only by /health and /db-status — not on every request.
    """
    global _connected

    # Fast path — startup ping already confirmed connection
    if _connected:
        return True

    # _client is None only if MongoClient() itself failed (bad URI etc.)
    if _client is None:
        return False

    # Live retry — handles the case where startup ping raced/timed out
    try:
        _client.admin.command("ping")
        _connected = True   # Cache success so subsequent calls are O(1)
        print("✅ [MongoDB] Reconnected on health check", flush=True)
        return True
    except Exception:
        return False
