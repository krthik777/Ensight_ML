import sqlite3
import yaml
import os
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to config.yaml in project root
            config_path = Path(__file__).parent.parent / 'config.yaml'
        
        self.config = self._load_config(config_path)
        self.db_path = self.config.get('database', {}).get('path', 'energy_data.db')
        self._init_db()

    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Error loading config: {e}. Using defaults.")
            return {}

    def _init_db(self):
        """Initialize the database with required tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS power_readings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        room_id TEXT NOT NULL,
                        power_value REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        except Exception as e:
            print(f"❌ Database initialization error: {e}")

    def get_recent_readings(self, room_id, limit=50):
        """
        Fetch the most recent power readings for a specific room.
        Returns a list of float values, ordered from oldest to newest.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT power_value 
                    FROM power_readings 
                    WHERE room_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (room_id, limit))
                
                rows = cursor.fetchall()
                
                # Rows are (value,), (value,), ... ordered DESC (newest first)
                # We need to reverse them to be chronological (oldest -> newest)
                readings = [r[0] for r in rows][::-1]
                
                return readings
        except Exception as e:
            print(f"❌ Error fetching readings: {e}")
            return []

    def insert_reading(self, room_id, power_value):
        """Insert a single reading (useful for testing/simulation)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO power_readings (room_id, power_value)
                    VALUES (?, ?)
                ''', (room_id, power_value))
                conn.commit()
                return True
        except Exception as e:
            print(f"❌ Error inserting reading: {e}")
            return False
