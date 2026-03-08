
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from src.models import get_model
from src.preprocessor import NILMPreprocessor

def evaluate_saved_model():
    print("📊 Evaluating Saved Model on Synthetic Data...")
    
    # Load config
    try:
        with open('models/training_config.json', 'r') as f:
            config = json.load(f)
        appliance_names = config['appliance_names']
        print(f"✅ Loaded config for {len(appliance_names)} appliances")
    except FileNotFoundError:
        print("❌ Config file not found. Please train the model first.")
        return

    # Generate synthetic testing data
    print("🧪 Generating synthetic test data...")
    n_samples = 2000
    seq_len = config.get('sequence_length', 50)
    
    # Create random but structured data
    # Mains power (input)
    X_test = np.abs(np.random.randn(n_samples, seq_len, 1).astype(np.float32))
    # Appliance states (ground truth) - varying sparsity
    y_test = np.abs(np.random.randn(n_samples, len(appliance_names)).astype(np.float32))
    
    # 3. Load Model
    print("🔄 Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(
        'lstm', 
        input_size=1, 
        output_size=len(appliance_names), 
        hidden_size=128,  # Assuming default from train_complete_system lines 107
        num_layers=2      # Assuming default from train_complete_system lines 108
    ).to(device)
    
    try:
        model.load_state_dict(torch.load('models/best_classification_model.pth', map_location=device))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Try loading minimal model params if complete failed
        try:
             model = get_model('lstm', input_size=1, output_size=len(appliance_names), hidden_size=64, num_layers=1).to(device)
             model.load_state_dict(torch.load('models/best_classification_model.pth', map_location=device))
             print("✅ Model loaded (minimal config)")
        except Exception as e2:
             print(f"❌ Error loading minimal model: {e2}")
             return

    model.eval()
    
    # 4. Run Prediction
    print("🔮 Running predictions...")
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test).float().to(device)
        predictions = model(X_tensor).cpu().numpy()
        
    # 5. Calculate Metrics
    print("📉 Calculating metrics...")
    metrics = {}
    
    # Binary classification threshold
    threshold = 0.5
    
    # Create results dataframe
    results = []
    
    for i, appliance in enumerate(appliance_names):
        # Ground truth
        actual = y_test[:, i]
        # Predicted
        pred = predictions[:, i]
        
        # Binarize for classification metrics
        actual_binary = (actual > 0.1).astype(int)
        pred_binary = (pred > 0.1).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(actual_binary, pred_binary)
        prec = precision_score(actual_binary, pred_binary, zero_division=0)
        rec = recall_score(actual_binary, pred_binary, zero_division=0)
        f1 = f1_score(actual_binary, pred_binary, zero_division=0)
        
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        
        metrics[appliance] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'MSE': mse,
            'MAE': mae
        }
        
        results.append({
            'Appliance': appliance,
            'Metric': 'Accuracy',
            'Value': acc
        })
        results.append({
            'Appliance': appliance,
            'Metric': 'F1-Score',
            'Value': f1
        })

    # 6. Generate Visualizations
    print("🎨 Generating visualizations...")
    Path("evaluation_results").mkdir(exist_ok=True)
    
    # Create a comprehensive plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Metrics Comparison
    df_results = pd.DataFrame(results)
    sns.barplot(data=df_results, x='Appliance', y='Value', hue='Metric', ax=axes[0])
    axes[0].set_title('Model Performance Metrics by Appliance')
    axes[0].set_ylim(0, 1.1)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Sample Prediction vs Actual (for first appliance)
    app_idx = 0
    sample_range = range(100)
    axes[1].plot(y_test[sample_range, app_idx], label='Actual', alpha=0.7)
    axes[1].plot(predictions[sample_range, app_idx], label='Predicted', alpha=0.7)
    axes[1].set_title(f'Sample Predictions: {appliance_names[app_idx]}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/real_model_performance.png')
    print("✅ Saved plot: evaluation_results/real_model_performance.png")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_df.to_csv('evaluation_results/real_model_metrics.csv')
    print("✅ Saved metrics: evaluation_results/real_model_metrics.csv")
    
    print("\n📊 Summary Metrics:")
    print(metrics_df[['Accuracy', 'F1-Score', 'MAE']])

if __name__ == "__main__":
    evaluate_saved_model()
