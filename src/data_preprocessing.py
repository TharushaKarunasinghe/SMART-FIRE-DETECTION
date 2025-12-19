"""
Data preprocessing for fire detection system
Loads, cleans, and combines multiple datasets
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class DataPreprocessor:
    def __init__(self):
        self.df_combined = None
        
    def load_smoke_detection_dataset(self):
        """Load Kaggle Smoke Detection Dataset"""
        print("Loading Smoke Detection Dataset...")
        
        try:
            df = pd.read_csv(os.path.join(RAW_DATA_DIR, SMOKE_DATASET))
            print(f"Loaded {len(df)} records")
            print(f"Columns: {df.columns.tolist()}")
            
            # Rename columns to standardize
            # Adjust based on actual column names
            column_mapping = {
                'Temperature[C]': 'temperature',
                'Humidity[%]': 'humidity',
                'TVOC[ppb]': 'gas',
                'eCO2[ppm]': 'smoke',
                'Fire Alarm': 'fire_alarm'
            }
            
            df = df.rename(columns=column_mapping)
            return df
            
        except FileNotFoundError:
            print(f"Error: {SMOKE_DATASET} not found in {RAW_DATA_DIR}")
            print("Please download the dataset and place it in data/raw/")
            return None
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def load_environmental_dataset(self):
        """Load Environmental Sensor Dataset"""
        print("\nLoading Environmental Sensor Dataset...")
        
        try:
            df = pd.read_csv(os.path.join(RAW_DATA_DIR, ENV_DATASET))
            print(f"Loaded {len(df)} records")
            print(f"Columns: {df.columns.tolist()}")
            
            # Standardize column names
            column_mapping = {
                'temp': 'temperature',
                'humid': 'humidity',
                'smoke': 'smoke',
                'co': 'gas'
            }
            
            df = df.rename(columns=column_mapping)
            return df
            
        except FileNotFoundError:
            print(f"Note: {ENV_DATASET} not found. Continuing with available data...")
            return None
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def clean_data(self, df):
        """Clean and handle missing values"""
        print("\nCleaning data...")
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_len - len(df)} duplicate rows")
        
        # Handle missing values
        print(f"Missing values before cleaning:\n{df.isnull().sum()}")
        df = df.dropna()
        print(f"Rows after removing NaN: {len(df)}")
        
        # Remove outliers (optional - can be adjusted)
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"Rows after outlier removal: {len(df)}")
        return df
    
    def create_warning_levels(self, df):
        """Create 5-level warning system based on sensor readings"""
        print("\nCreating warning level labels...")
        
        def assign_warning_level(row):
            temp = row['temperature']
            humidity = row['humidity']
            smoke = row['smoke']
            gas = row['gas']
            
            # Level 5: Emergency - Clear fire conditions
            if temp > 50 or smoke > 400 or gas > 500:
                return 4
            
            # Level 4: Warning - High risk
            elif temp > 40 or smoke > 250 or gas > 350:
                return 3
            
            # Level 3: Caution - Moderate concern
            elif temp > 32 or smoke > 150 or gas > 200:
                return 2
            
            # Level 2: Watch - Slight elevation
            elif temp > 28 or smoke > 80 or gas > 120:
                return 1
            
            # Level 1: All Clear
            else:
                return 0
        
        df['warning_level'] = df.apply(assign_warning_level, axis=1)
        
        # Print distribution
        print("\nWarning level distribution:")
        for level, name in WARNING_LEVELS.items():
            count = len(df[df['warning_level'] == level])
            percentage = (count / len(df)) * 100
            print(f"Level {level} ({name}): {count} ({percentage:.2f}%)")
        
        return df
    
    def combine_datasets(self, *dataframes):
        """Combine multiple datasets"""
        print("\nCombining datasets...")
        
        valid_dfs = [df for df in dataframes if df is not None]
        
        if not valid_dfs:
            print("Error: No valid datasets to combine")
            return None
        
        combined = pd.concat(valid_dfs, ignore_index=True)
        print(f"Combined dataset size: {len(combined)} records")
        
        return combined
    
    def save_processed_data(self, df, filename="processed_fire_data.csv"):
        """Save processed data"""
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"\nProcessed data saved to: {filepath}")
        print(f"Final dataset shape: {df.shape}")

def main():
    """Main preprocessing pipeline"""
    print("="*60)
    print("SMART FIRE DETECTION - DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    
    # Load datasets
    df_smoke = preprocessor.load_smoke_detection_dataset()
    df_env = preprocessor.load_environmental_dataset()
    
    # Check if at least one dataset loaded successfully
    if df_smoke is None and df_env is None:
        print("\nError: No datasets could be loaded. Please check:")
        print("1. Datasets are downloaded from Kaggle")
        print("2. Files are placed in data/raw/ directory")
        print("3. Filenames match config.py settings")
        return
    
    # Clean individual datasets
    datasets = []
    if df_smoke is not None:
        df_smoke_clean = preprocessor.clean_data(df_smoke)
        datasets.append(df_smoke_clean)
    
    if df_env is not None:
        df_env_clean = preprocessor.clean_data(df_env)
        datasets.append(df_env_clean)
    
    # Combine datasets
    df_combined = preprocessor.combine_datasets(*datasets)
    
    # Ensure we have required columns
    if not all(col in df_combined.columns for col in FEATURE_COLUMNS):
        print(f"\nError: Missing required columns. Required: {FEATURE_COLUMNS}")
        print(f"Available: {df_combined.columns.tolist()}")
        return
    
    # Create warning levels
    df_combined = preprocessor.create_warning_levels(df_combined)
    
    # Select final columns
    final_columns = FEATURE_COLUMNS + ['warning_level']
    df_final = df_combined[final_columns]
    
    # Sample if too large
    if len(df_final) > TARGET_SAMPLES:
        df_final = df_final.sample(n=TARGET_SAMPLES, random_state=RANDOM_STATE)
        print(f"\nSampled {TARGET_SAMPLES} records for training")
    
    # Save processed data
    preprocessor.save_processed_data(df_final)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nNext step: Run 'python src/train_model.py' to train the model")

if __name__ == "__main__":
    main()
