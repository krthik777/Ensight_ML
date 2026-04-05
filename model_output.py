# model_output.py
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class ModelOutputDisplay:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/predict"
        self.health_url = f"{base_url}/health"
    
    def check_api_health(self):
        """Check if API is running"""
        try:
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                print("✅ API is running and healthy")
                return True
            else:
                print("❌ API returned non-200 status")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ API is not running. Please start the API first with: python start_api.py")
            return False
        except Exception as e:
            print(f"❌ Error connecting to API: {e}")
            return False
    
    def generate_realistic_inputs(self, num_samples=5):
        """Generate realistic power consumption inputs"""
        
        # Realistic power ranges for Indian households (in watts)
        appliance_ranges = {
            'mains': (200, 5000),           # Total household power
            'air_conditioner': (800, 3000), # AC power consumption
            'fridge': (100, 300),          # Refrigerator
            'television': (50, 200),       # TV
            'washing_machine': (300, 800), # Washing machine
            'water_motor': (400, 1500),    # Water pump
            'kitchen_outlets': (100, 2000), # Kitchen appliances
            'laptop_computer': (30, 100),  # Laptop/computer
            'iron': (800, 1500),           # Iron
            'water_filter': (25, 50)       # Water purifier
        }
        
        samples = []
        for i in range(num_samples):
            # Generate realistic timestamp (last 24 hours)
            timestamp = datetime.now() - timedelta(hours=24-i*6)
            
            # Create sample with realistic correlations
            sample = {}
            
            # Start with some base appliances
            ac_power = np.random.randint(0, 2500) if np.random.random() > 0.7 else 0  # AC often off
            fridge_power = np.random.randint(100, 200)  # Fridge cycles but always some consumption
            tv_power = np.random.randint(0, 150) if np.random.random() > 0.5 else 0
            kitchen_power = np.random.randint(0, 800) if np.random.random() > 0.6 else 0
            
            # Calculate realistic mains (sum of active appliances + some noise)
            active_appliances = [ac_power, fridge_power, tv_power, kitchen_power]
            mains_power = sum(active_appliances) + np.random.randint(0, 200)
            
            sample = {
                'mains': float(mains_power),
                'air_conditioner': float(ac_power),
                'fridge': float(fridge_power),
                'television': float(tv_power),
                'washing_machine': float(np.random.randint(0, 600) if np.random.random() > 0.8 else 0),
                'water_motor': float(np.random.randint(0, 1000) if np.random.random() > 0.9 else 0),
                'kitchen_outlets': float(kitchen_power),
                'laptop_computer': float(np.random.randint(0, 80) if np.random.random() > 0.6 else 0),
                'iron': float(np.random.randint(0, 1200) if np.random.random() > 0.9 else 0),
                'water_filter': float(np.random.randint(25, 50)),
                'timestamp': timestamp.isoformat()
            }
            samples.append(sample)
        
        return samples
    
    def display_prediction_results(self, input_data, prediction):
        """Display formatted prediction results"""
        print("\n" + "="*80)
        print("🤖 MODEL PREDICTION RESULTS")
        print("="*80)
        
        # Display input data
        print("\n📊 INPUT DATA:")
        print("-" * 40)
        for key, value in input_data.items():
            if key != 'timestamp':
                print(f"  {key.replace('_', ' ').title():<20}: {value:>8.1f} W")
            else:
                print(f"  {'Timestamp':<20}: {value}")
        
        # Display predictions
        print("\n🎯 APPLIANCE PREDICTIONS:")
        print("-" * 40)
        
        if 'predictions' in prediction:
            for appliance, pred_value in prediction['predictions'].items():
                actual_value = input_data.get(appliance, 0)
                error = abs(pred_value - actual_value) if actual_value > 0 else 0
                error_percent = (error / actual_value * 100) if actual_value > 0 else 0
                
                status = "✅" if error_percent < 20 else "⚠️" if error_percent < 50 else "❌"
                
                print(f"  {appliance.replace('_', ' ').title():<20}: {pred_value:>8.1f} W")
                if actual_value > 0:
                    print(f"    {'Actual':<18}: {actual_value:>8.1f} W")
                    print(f"    {'Error':<18}: {error:>8.1f} W ({error_percent:.1f}%) {status}")
                print()
        
        # Display summary metrics if available
        if 'confidence' in prediction:
            print(f"\n📈 PREDICTION CONFIDENCE: {prediction['confidence']:.1%}")
        
        if 'total_consumption' in prediction:
            print(f"💡 TOTAL PREDICTED CONSUMPTION: {prediction['total_consumption']:.1f} W")
        
        # Display inference time if available
        if 'inference_time' in prediction:
            print(f"⏱️  INFERENCE TIME: {prediction['inference_time']:.3f} seconds")
        
        print("="*80)
    
    def run_demo(self, num_samples=3):
        """Run a demo with multiple realistic samples"""
        print("🚀 Starting Model Output Display")
        print("🔍 Checking API connectivity...")
        
        if not self.check_api_health():
            return
        
        print(f"\n🎲 Generating {num_samples} realistic power consumption samples...")
        samples = self.generate_realistic_inputs(num_samples)
        
        successful_predictions = 0
        
        for i, sample in enumerate(samples, 1):
            print(f"\n{'='*60}")
            print(f"📋 SAMPLE {i}/{num_samples}")
            print(f"{'='*60}")
            
            try:
                # Send prediction request
                print("🔄 Sending request to model...")
                start_time = time.time()
                
                response = requests.post(
                    self.api_url,
                    json=sample,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                inference_time = time.time() - start_time
                
                if response.status_code == 200:
                    prediction = response.json()
                    prediction['inference_time'] = inference_time
                    self.display_prediction_results(sample, prediction)
                    successful_predictions += 1
                else:
                    print(f"❌ API Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"❌ Request failed: {e}")
            
            # Small delay between samples
            if i < len(samples):
                print("\n⏳ Preparing next sample...")
                time.sleep(2)
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 DEMO SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful predictions: {successful_predictions}/{num_samples}")
        print(f"🎯 Success rate: {(successful_predictions/num_samples)*100:.1f}%")
        
        if successful_predictions > 0:
            print("🎉 Model is working correctly!")
        else:
            print("❌ No successful predictions. Check API and model.")
        
        print(f"{'='*60}")

def main():
    """Main function to run the model output display"""
    display = ModelOutputDisplay()
    
    print("🤖 NILM Model Output Display")
    print("This script demonstrates realistic model predictions")
    print("Make sure your API is running on http://localhost:5000")
    print()
    
    try:
        display.run_demo(num_samples=3)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()