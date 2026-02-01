# train_complete_system.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import os

Path("models").mkdir(exist_ok=True)
Path("saved_models").mkdir(exist_ok=True)
Path("api").mkdir(exist_ok=True)

from src.models import get_model
from src.preprocessor import NILMPreprocessor

class ClassificationTrainer:
    def __init__(self, model_type='lstm', input_size=1, num_appliances=9,
                 hidden_size=128, num_layers=2, lr=1e-3, batch_size=32, dropout=0.3):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(model_type, input_size=input_size, output_size=num_appliances,
                              hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(self.device)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)

    def train(self, X, y, epochs=30, validation_split=0.2):
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        
        dataset_size = len(X_tensor)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.scheduler.step(avg_val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1:02d}/{epochs}] | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model('models/best_classification_model.pth')

        print(f"✅ Training completed! Best val loss: {best_val_loss:.6f}")
        return {'train_loss': avg_train_loss, 'val_loss': avg_val_loss}

    def save_model(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

def main():
    print("🏠 NILM COMPLETE TRAINING SYSTEM")
    print("="*50)
    
    # Load and preprocess data
    preprocessor = NILMPreprocessor()
    merged_df, appliance_cols = preprocessor.load_all_data()
    
    # Normalize
    all_power_cols = ['mains_W'] + appliance_cols
    normalized_df, max_vals = preprocessor.normalize_data(merged_df, all_power_cols)
    
    # Create sequences
    X, y = preprocessor.create_sequences(normalized_df, appliance_cols, seq_len=50, sample_rate=5)
    
    print(f"📊 Training on {X.shape[0]} sequences, {len(appliance_cols)} appliances")
    
    # Train model
    trainer = ClassificationTrainer(
        model_type='lstm',
        input_size=1,
        num_appliances=len(appliance_cols),
        hidden_size=128,
        num_layers=2,
        lr=1e-3,
        batch_size=32,
        dropout=0.3
    )
    
    history = trainer.train(X, y, epochs=30, validation_split=0.2)
    
    # Save configuration
    config = {
        'appliance_names': [col.replace('_W', '') for col in appliance_cols],
        'max_values': max_vals,
        'sequence_length': 50,
        'training_date': datetime.now().isoformat()
    }
    
    with open('models/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("🎉 Training completed!")
    print("💾 Model saved: models/best_classification_model.pth")
    print("📋 Config saved: models/training_config.json")

if __name__ == "__main__":
    main()