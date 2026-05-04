"""
run.py — Root-level launcher for the EnSight AI Model Engine.
Run from the project root: python run.py
"""
import sys
import os

# Load .env BEFORE any other imports so MONGO_URI is available
# when api/mongo_client.py initialises at import time.
# On Render this is a no-op (env vars come from the dashboard).
try:
    from dotenv import load_dotenv
    load_dotenv()          # reads .env from project root
    print("📄 Loaded .env file")
except ImportError:
    pass   # python-dotenv not installed — env vars must be set another way

# Ensure project root is always in sys.path so all modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print("🚀 Starting EnSight AI Model Engine...")
    print(f"🌐 Server:  http://localhost:{port}")
    print("🔌 Endpoints available:")
    print("   GET  /                         - Service info")
    print("   GET  /health                   - Health check")
    print("   POST /detect-appliances        - Active appliances + confidence")
    print("   POST /predict-appliance-power  - Per-appliance power (W)")
    print("   POST /predict-cost             - Monthly KSEB cost from daily kWh")
    print("   POST /predict-month-end        - Month-end unit & cost forecast")
    print("   POST /detect-anomaly           - Power spike / anomaly detection")
    print("-" * 55)
    app.run(debug=False, host='0.0.0.0', port=port)
