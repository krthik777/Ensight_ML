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

from src.models import get_model, MultiOutputLSTM, ConvLSTM


class ContinuousNILMTrainer:
    """Trainer for multi-appliance NILM using various neural architectures."""

    def __init__(self, model_type='lstm', input_size=1, num_appliances=9,
                 hidden_size=128, num_layers=2, lr=1e-3, batch_size=64, 
                 device=None, model_config=None):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_appliances = num_appliances
        self.batch_size = batch_size
        self.model_type = model_type

        print(f"🚀 Initializing {model_type.upper()} model on {self.device}...")
        
        # Model configuration
        model_config = model_config or {}
        self.model = get_model(
            model_type, 
            input_size=input_size, 
            output_size=num_appliances,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **model_config
        ).to(self.device)

        # Loss function - use SmoothL1Loss for better stability than MSE
        self.criterion = nn.SmoothL1Loss()
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        print(f"✅ Model initialized: {self.model.__class__.__name__}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        return epoch_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
                
        return val_loss / len(val_loader)

    def train(self, X, y, epochs=50, validation_split=0.1, early_stopping_patience=10):
        """Main training loop"""
        print(f"\n🎯 Starting training for {epochs} epochs...")
        print(f"📊 Dataset: {X.shape[0]:,} sequences, {X.shape[2]} features")
        print(f"🎯 Target: {y.shape[1]} appliances")
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)

        # Train-validation split
        dataset_size = len(X_tensor)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = random_split(
            TensorDataset(X_tensor, y_tensor), [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        print(f"📈 Training on {train_size:,} samples, validating on {val_size:,} samples")
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_times'].append(time.time() - epoch_start_time)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"📅 Epoch [{epoch+1:03d}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"LR: {current_lr:.2e} | Time: {self.history['epoch_times'][-1]:.1f}s")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.load_model('best_model.pth')
        print(f"✅ Training completed! Best validation loss: {best_val_loss:.4f}")
        
        return self.history

    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        
        print(f"📊 Evaluation Metrics:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {np.sqrt(mse):.4f}")
        
        return {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse)}

    def save_model(self, filepath):
        """Save model and training history"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'model_type': self.model_type,
            'num_appliances': self.num_appliances,
            'model_config': {
                'hidden_size': getattr(self.model, 'hidden_size', None),
                'num_layers': getattr(self.model, 'num_layers', None)
            }
        }
        
        torch.save(checkpoint, Path(__file__).parent.parent / 'models' / filepath)
        print(f"💾 Model saved: {filepath}")

    def load_model(self, filepath):
        """Load model from checkpoint"""
        checkpoint_path = Path(__file__).parent.parent / 'models' / filepath
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.history = checkpoint['history']
            print(f"📥 Model loaded: {filepath}")
        else:
            print(f"⚠️ No checkpoint found at {checkpoint_path}")

    def continuous_update(self, new_X, new_y):
        """Online retraining with a small new batch"""
        self.model.train()
        new_X_tensor = torch.from_numpy(new_X).float().to(self.device)
        new_y_tensor = torch.from_numpy(new_y).float().to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(new_X_tensor)
        loss = self.criterion(outputs, new_y_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def main():
    """Main training function"""
    from src.preprocessor import NILMPreprocessor
    
    print("=" * 60)
    print("🏠 NILM Model Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'model_type': 'lstm',  # 'lstm', 'convlstm', 'attention'
        'seq_len': 50,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-3,
        'hidden_size': 128,
        'num_layers': 2,
        'sample_rate': 10  # Reduce dataset size for faster training
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Initialize preprocessor
    preprocessor = NILMPreprocessor()
    
    # Load and preprocess data
    print("\n📥 Loading data...")
    merged_df, appliance_cols = preprocessor.load_all_data()
    
    # Normalize data
    all_power_cols = ['mains_W'] + appliance_cols
    normalized_df, max_vals = preprocessor.normalize_data(merged_df, all_power_cols)
    
    # Create sequences
    X, y = preprocessor.create_sequences(
        normalized_df, appliance_cols, 
        seq_len=config['seq_len'], 
        sample_rate=config['sample_rate']
    )

    print(f"\n🎯 Training Setup:")
    print(f"   Sequences: {X.shape[0]:,}")
    print(f"   Sequence length: {X.shape[1]}")
    print(f"   Appliances: {len(appliance_cols)}")
    print(f"   Appliance names: {[col.replace('_W', '') for col in appliance_cols]}")
    
    # Initialize trainer
    trainer = ContinuousNILMTrainer(
        model_type=config['model_type'],
        input_size=1,
        num_appliances=y.shape[1],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        lr=config['learning_rate'],
        batch_size=config['batch_size']
    )
    
    # Train model
    history = trainer.train(
        X, y, 
        epochs=config['epochs'],
        validation_split=0.1,
        early_stopping_patience=15
    )
    
    # Evaluate model
    print("\n📊 Final Evaluation:")
    trainer.evaluate(X[:1000], y[:1000])  # Evaluate on subset
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"nilm_{config['model_type']}_{timestamp}.pth"
    trainer.save_model(model_filename)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"💾 Model saved as: {model_filename}")
    
    return trainer, history


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Run training
    trainer, history = main()