# train_minimal.py
import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from src.models import get_model

# Create models directory
Path("models").mkdir(exist_ok=True)

def train_minimal():
    print("⚡ CREATING MINIMAL TRAINED MODEL...")
    
    # Create tiny synthetic dataset
    n_sequences = 1000
    seq_len = 50
    n_appliances = 9
    
    # Generate realistic data patterns
    X = np.random.randn(n_sequences, seq_len, 1).astype(np.float32) * 0.1 + 0.5
    y = np.abs(np.random.randn(n_sequences, n_appliances).astype(np.float32) * 0.1)
    
    # Simple model
    model = get_model('lstm', input_size=1, output_size=n_appliances, hidden_size=64, num_layers=1)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training (5 epochs)
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/5] Loss: {loss.item():.6f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/best_classification_model.pth')
    
    # Create config with realistic appliance names
    config = {
        'appliance_names': [
            'air_conditioner', 'fridge', 'iron', 'kitchen_outlets', 
            'laptop_computer', 'television', 'washing_machine', 
            'water_filter', 'water_motor'
        ],
        'max_values': {
            'mains_W': 5000.0,
            'air_conditioner_W': 2000.0,
            'fridge_W': 300.0,
            'iron_W': 1500.0,
            'kitchen_outlets_W': 1500.0,
            'laptop_computer_W': 100.0,
            'television_W': 200.0,
            'washing_machine_W': 500.0,
            'water_filter_W': 100.0,
            'water_motor_W': 1500.0
        },
        'sequence_length': 50,
        'training_date': datetime.now().isoformat(),
        'note': 'Minimal model for API testing'
    }
    
    with open('models/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Minimal model created!")
    print("💾 Model: models/best_classification_model.pth")
    print("📋 Config: models/training_config.json")

if __name__ == "__main__":
    train_minimal()