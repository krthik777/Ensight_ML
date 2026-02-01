
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.db_manager import DatabaseManager
from api.nilm_integration import predict_appliances

def test_db_integration():
    print("Testing Database Integration...")
    
    # 1. Setup DB and Data
    db = DatabaseManager()
    room_id = "test_room_001"
    
    # Clear existing data for test room (mocking this by just inserting new fresh data)
    # Ideally should have a delete method, but inserts are fine for this check
    
    print(f"Inserting 50 readings for {room_id}...")
    for i in range(50):
        # Insert a ramp pattern: 100, 200, ... 5000
        val = (i + 1) * 100
        db.insert_reading(room_id, val)
        
    # 2. Verify Data Fetching
    readings = db.get_recent_readings(room_id, limit=50)
    print(f"Fetched {len(readings)} readings.")
    assert len(readings) == 50, f"Expected 50 readings, got {len(readings)}"
    assert readings[-1] == 5000, f"Expected last reading 5000, got {readings[-1]}"
    print("✅ Data fetching verified.")

    # 3. Verify Prediction via Room ID
    print("Running prediction using Room ID...")
    result = predict_appliances(room_id=room_id)
    
    print(json.dumps(result, indent=2))
    
    assert result['totalPower'] > 0, "Expected non-zero total power"
    assert 'appliances' in result, "Expected appliances in result"
    
    print("✅ Prediction verified.")
    print("🎉 Test PASSED!")

if __name__ == "__main__":
    test_db_integration()
