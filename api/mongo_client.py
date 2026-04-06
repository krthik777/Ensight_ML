"""
api/mongo_client.py
--------------------
MongoDB client — ZERO hardcoded credentials.

MONGO_URI must be set as an environment variable:
  - Render  → Dashboard → Environment → MONGO_URI (secret)
  - Local   → .env file (never commit):
      MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/db?retryWrites=true&w=majority

NOTE: All startup prints use flush=True because Render runs Python in a
non-TTY pipe environment where stdout is block-buffered by default.
Set PYTHONUNBUFFERED=1 in Render env vars as an additional safeguard.
"""

import os
import sys
import yaml
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError

# ── Singletons ─────────────────────────────────────────────────────────────────
_client: "MongoClient | None" = None   # string annotation avoids Python 3.9 | issue
_db     = None
_connected: bool = False


def _load_mongo_uri() -> "tuple[str, str]":
    """
    Returns (full_uri, db_name).
    MONGO_URI env var is the authoritative source — never hardcoded here.
    """
    # ── env var (Render dashboard / .env) ─────────────────────────────────────
    uri = os.environ.get("MONGO_URI", "").strip()

    # ── fallback: config.yaml (should be blank uri in production) ─────────────
    if not uri:
        try:
            cfg_path = Path(__file__).parent.parent / "config.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            uri = cfg.get("mongodb", {}).get("uri", "").strip()
        except Exception:
            pass

    # ── db name ───────────────────────────────────────────────────────────────
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
        print("   Set it in: Render → Environment → MONGO_URI", flush=True, file=sys.stderr)
        print("   Format: mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/<db>?retryWrites=true&w=majority", flush=True, file=sys.stderr)
        return "", db_name

    # ── Append Atlas required options if missing ───────────────────────────────
    if "mongodb+srv" in uri and "?" not in uri:
        uri = f"{uri.rstrip('/')}/{db_name}?retryWrites=true&w=majority&appName=Cluster0"

    return uri, db_name


# ── Connect synchronously at module import ────────────────────────────────────
# flush=True on every print — critical in Render's buffered pipe environment.
# Without flush=True, logs only appear when the buffer fills or process exits.
try:
    _uri, _db_name = _load_mongo_uri()

    if _uri:
        safe_uri = _uri[:50] + "..." if len(_uri) > 50 else _uri
        print(f"🔌 [MongoDB] Connecting to Atlas → {_db_name}", flush=True)
        print(f"   URI (truncated): {safe_uri}", flush=True)

        _client = MongoClient(
            _uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,
            tls=True,
        )
        _db = _client[_db_name]

        # Synchronous ping — blocks until Atlas responds or times out
        _client.admin.command("ping")
        _connected = True
        print(f"✅ [MongoDB] Connected → {_db_name}", flush=True)

    else:
        print("⚠️  [MongoDB] Skipped — MONGO_URI not set. DB endpoints will return 503.", flush=True)

except ConfigurationError as e:
    print(f"❌ [MongoDB] Configuration error: {e}", flush=True, file=sys.stderr)
    print("   Check MONGO_URI format in Render Environment settings.", flush=True, file=sys.stderr)
    _client = None
    _db     = None

except ServerSelectionTimeoutError as e:
    print(f"❌ [MongoDB] Connection TIMEOUT after 10s: {e}", flush=True, file=sys.stderr)
    print("   Most likely cause: MongoDB Atlas IP Whitelist is blocking Render's IPs.", flush=True, file=sys.stderr)
    print("   Fix: Atlas → Network Access → Add IP Address → 0.0.0.0/0", flush=True, file=sys.stderr)
    _client    = None
    _db        = None
    _connected = False

except Exception as e:
    print(f"❌ [MongoDB] Unexpected error — {type(e).__name__}: {e}", flush=True, file=sys.stderr)
    _client    = None
    _db        = None
    _connected = False


# ── Public API ─────────────────────────────────────────────────────────────────

def get_db():
    """Returns the MongoDB database reference. None if not connected."""
    return _db


def is_connected() -> bool:
    """Live ping — used by /health and /db-status only."""
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
