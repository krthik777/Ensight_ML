# check_data_quality.py
import pandas as pd
from pathlib import Path

def check_data_quality():
    print("🔍 DATA QUALITY CHECK")
    print("="*40)
    
    processed_files = list(Path('data/processed').glob('*_processed.csv'))
    
    for file in processed_files:
        print(f"\n📊 {file.name}:")
        df = pd.read_csv(file, nrows=5)  # Just check first few rows
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if 'W' in df.columns:
            print(f"  Power range: {df['W'].min():.3f} - {df['W'].max():.3f}W")
        print(f"  Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

if __name__ == "__main__":
    check_data_quality()