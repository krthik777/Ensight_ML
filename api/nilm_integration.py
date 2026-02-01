# api/nilm_integration.py - Converted to module
import torch
import numpy as np
import json
from datetime import datetime
from collections import deque
import os
from src.models import get_model
from api.db_manager import DatabaseManager

class NILMIntegration:
    def __init__(self):
        self.model = None
        self.config = None
        self.sequence_buffer = deque(maxlen=50)
        self.db_manager = DatabaseManager()
        self.load_model()
    
    def load_model(self):
        """Load the trained NILM model"""
        try:
            with open('models/training_config.json', 'r') as f:
                self.config = json.load(f)
            
            self.model = get_model('lstm', input_size=1, output_size=len(self.config['appliance_names']))
            
            try:
                self.model.load_state_dict(torch.load('models/best_classification_model.pth', map_location='cpu'))
                print("✅ NILM Model loaded successfully")
            except Exception as e:
                print(f"⚠️ Model loading issue: {e}")
                print("Using initialized weights for demonstration")
                
            self.model.eval()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Create default config
            self.config = {
                'appliance_names': [
                    'air_conditioner', 'fridge', 'television', 'washing_machine', 
                    'laptop_computer', 'kitchen_outlets', 'iron', 'water_filter', 'water_motor'
                ],
                'max_values': {
                    'mains_W': 10000.0,
                    'air_conditioner_W': 3000.0,
                    'fridge_W': 300.0,
                    'television_W': 200.0,
                    'washing_machine_W': 800.0,
                    'laptop_computer_W': 100.0,
                    'kitchen_outlets_W': 1500.0,
                    'iron_W': 1500.0,
                    'water_filter_W': 100.0,
                    'water_motor_W': 1500.0
                }
            }
    
    def predict_appliances(self, mains_sequence=None, room_id=None):
        """Predict appliances from mains sequence (or DB) - returns prediction dict only"""
        
        # If room_id is provided, try to fetch from DB
        if room_id:
            db_sequence = self.db_manager.get_recent_readings(room_id, limit=50)
            if db_sequence and len(db_sequence) > 0:
                print(f"📊 Fetched {len(db_sequence)} readings for Room {room_id} from DB")
                if len(db_sequence) < 50:
                    # Pad with last known value if we have at least some data
                     db_sequence = db_sequence + [db_sequence[-1]] * (50 - len(db_sequence))
                mains_sequence = db_sequence
            elif mains_sequence is None:
                print(f"⚠️ No data found for Room {room_id} and no sequence provided.")
                return self.get_fallback_prediction([0]*50)

        if not mains_sequence:
             return self.get_fallback_prediction([0]*50)

        if self.model is None:
            return self.get_fallback_prediction(mains_sequence)
        
        try:
            if len(mains_sequence) != 50:
                # Pad or truncate to 50 if needed
                if len(mains_sequence) > 50:
                    mains_sequence = mains_sequence[-50:]
                else:
                    mains_sequence = mains_sequence + [mains_sequence[-1]] * (50 - len(mains_sequence))
            
            # Convert and normalize
            mains_seq = np.array(mains_sequence, dtype=np.float32)
            mains_normalized = mains_seq / self.config['max_values']['mains_W']
            
            # Reshape for model
            input_tensor = torch.from_numpy(mains_normalized.reshape(1, 50, 1)).float()
            
            # Predict
            with torch.no_grad():
                prediction = self.model(input_tensor).numpy()[0]
            
            # Convert to API format
            appliances = {}
            total_power = 0
            active_appliances = []
            
            for i, appliance_name in enumerate(self.config['appliance_names']):
                power = prediction[i] * self.config['max_values'][f"{appliance_name}_W"]
                power = max(0, float(power))
                
                state = "ON" if power > 15 else "OFF"
                
                appliances[appliance_name] = {
                    'power': power,
                    'state': state
                }
                
                if state == "ON":
                    active_appliances.append(appliance_name)
                    total_power += power
            
            confidence = min(0.95, np.max(prediction) * 2)
            
            return {
                'appliances': appliances,
                'totalPower': float(total_power),
                'confidence': float(confidence),
                'activeAppliances': active_appliances,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self.get_fallback_prediction(mains_sequence)
    
    def get_fallback_prediction(self, mains_sequence):
        """Fallback prediction"""
        avg_power = np.mean(mains_sequence)
        
        appliances = {}
        active_appliances = []
        total_power = 0
        
        appliance_rules = [
            ('air_conditioner', 1000, 0.4, 0.6),
            ('fridge', 100, 0.8, 0.1),
            ('television', 200, 0.3, 0.2),
            ('washing_machine', 400, 0.2, 0.3),
            ('laptop_computer', 80, 0.5, 0.1),
            ('kitchen_outlets', 300, 0.4, 0.2),
            ('iron', 800, 0.1, 0.8),
            ('water_filter', 50, 0.7, 0.05),
            ('water_motor', 600, 0.1, 0.5)
        ]
        
        for appliance, threshold, base_prob, power_ratio in appliance_rules:
            is_on = (avg_power > threshold and 
                    np.random.random() < base_prob + (avg_power / 5000 * 0.2))
            
            if is_on:
                power = avg_power * power_ratio + np.random.normal(0, threshold * 0.1)
                power = max(10, power)
                state = "ON"
                active_appliances.append(appliance)
                total_power += power
            else:
                power = 0
                state = "OFF"
            
            appliances[appliance] = {
                'power': float(power),
                'state': state
            }
        
        return {
            'appliances': appliances,
            'totalPower': float(total_power),
            'confidence': 0.5,
            'activeAppliances': active_appliances,
            'timestamp': datetime.now().isoformat(),
            'note': 'Fallback prediction'
        }

# Create global instance
nilm_predictor = NILMIntegration()

def predict_appliances(mains_sequence=None, room_id=None):
    """Main function to call from other modules"""
    return nilm_predictor.predict_appliances(mains_sequence, room_id)