# src/preprocessor.py
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings
import sys

warnings.filterwarnings('ignore')
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class NILMPreprocessor:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = project_root / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = project_root / "data/processed"
        self.sampling_rate = self.config['data']['sampling_rate']
        self.file_mappings = self.config['file_mappings']

    def load_file(self, file_number):
        file_path = self.data_path / f"{file_number}.csv"
        if not file_path.exists():
            return None, None
        
        df = pd.read_csv(file_path, na_values=['\\N', 'NULL', 'NaN'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        numeric_cols = ['W', 'VAR', 'VA', 'f', 'V', 'PF', 'A']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        appliance_type = self.file_mappings.get(str(file_number), f"unknown_{file_number}")
        return df, appliance_type

    def identify_real_mains(self):
        mains_files = []
        for fnum, appliance_type in self.file_mappings.items():
            if appliance_type == 'mains':
                mains_files.append(fnum)
        
        for fnum in mains_files:
            df, _ = self.load_file(fnum)
            if df is not None:
                max_power = df['W'].abs().max()
                unique_values = df['W'].nunique()
                if max_power > 10 and unique_values > 100:
                    return fnum
        return mains_files[-1] if mains_files else None

    def load_all_data(self):
        """Load all appliances + mains data from processed folder"""
        all_data = {}

        print("Loading data from processed folder...")

        # Look for processed files
        processed_files = list(self.data_path.glob("*_processed.csv"))

        if not processed_files:
            raise ValueError(f"No processed files found in {self.data_path}")

        for file_path in processed_files:
            appliance_name = file_path.stem.replace('_processed', '')

            try:
                df = pd.read_csv(file_path)
                # Try to parse timestamp flexibly
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                else:
                    raise KeyError('timestamp')

                # Drop rows with invalid timestamps
                df = df.dropna(subset=['timestamp'])

                # Handle different column structures
                if 'W' in df.columns:
                    all_data[appliance_name] = df
                    try:
                        print(f"✅ Loaded {appliance_name}: {len(df)} rows, power: {df['W'].min():.1f}-{df['W'].max():.1f}W")
                    except Exception:
                        print(f"✅ Loaded {appliance_name}: {len(df)} rows")
                else:
                    print(f"⚠️ Skipping {appliance_name}: No 'W' column found")

            except Exception as e:
                print(f"❌ Error loading {appliance_name}: {e}")

        if 'mains' not in all_data:
            raise ValueError("No mains data found in processed files!")

        # Use mains as reference timeline
        mains_df = all_data['mains'].sort_values('timestamp')
        common_start = mains_df['timestamp'].min()
        common_end = mains_df['timestamp'].max()

        # Create aligned dataframe starting with mains
        merged_df = mains_df[['timestamp', 'W']].copy()
        merged_df = merged_df.rename(columns={'W': 'mains_W'})

        # Merge appliance data
        appliance_cols = []
        for appliance, df in all_data.items():
            if appliance == 'mains':
                continue

            appliance_df = df[['timestamp', 'W']].rename(columns={'W': f'{appliance}_W'})
            # Ensure both are sorted for merge_asof
            merged_df = pd.merge_asof(
                merged_df.sort_values('timestamp'),
                appliance_df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
            appliance_cols.append(f'{appliance}_W')
            print(f"🔗 Aligned {appliance} to mains timeline")

        # Fill missing appliance values with 0 (appliance is off)
        if appliance_cols:
            merged_df[appliance_cols] = merged_df[appliance_cols].fillna(0)

        print(f"\n📈 Final merged data shape: {merged_df.shape}")
        print(f"🏷️ Columns: {list(merged_df.columns)}")
        print(f"⏰ Time period: {common_start} to {common_end}")
        try:
            print(f"⏱️ Duration: {(common_end - common_start).total_seconds() / 3600:.2f} hours")
        except Exception:
            pass

        return merged_df, appliance_cols

    def normalize_data(self, df, columns):
        df_norm = df.copy()
        max_vals = {}
        for col in columns:
            max_val = max(df[col].max(), 1.0)
            max_vals[col] = max_val
            df_norm[col] = df[col] / max_val
        return df_norm, max_vals

    def create_sequences(self, df, target_columns, seq_len=50, sample_rate=1):
        mains_values = df['mains_W'].values
        target_values = df[target_columns].values
        
        if sample_rate > 1:
            mains_values = mains_values[::sample_rate]
            target_values = target_values[::sample_rate]
        
        sequences_X, sequences_y = [], []
        for i in range(len(mains_values) - seq_len):
            sequences_X.append(mains_values[i:i+seq_len].reshape(-1, 1))
            sequences_y.append(target_values[i + seq_len - 1])
        
        return np.array(sequences_X, dtype=np.float32), np.array(sequences_y, dtype=np.float32)