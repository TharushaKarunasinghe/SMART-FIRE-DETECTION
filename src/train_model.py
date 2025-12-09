"""
Train machine learning model for fire detection
Trains Decision Tree and Random Forest models
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class FireDetectionTrainer:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.dt_model = None
        self.rf_model = None
    
    def load_processed_data(self):
        """Load preprocessed data"""
        print("Loading processed data...")
        
        filepath = os.path.join(PROCESSED_DATA_DIR, "processed_fire_data.csv")
        
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} records")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except FileNotFoundError:
            print(f"Error: Processed data not found at {filepath}")
            print("Please run 'python src/data_preprocessing.py' first")
            return None
    
    def prepare_features(self, df):
        """Prepare features and labels"""
        print("\nPreparing features and labels...")
        
        X = df[FEATURE_COLUMNS]
        y = df['warning_level']
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y):
        """Split data into training and testing sets"""
        print("\nSplitting data...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Testing set: {len(self.X_test)} samples")
    
    def normalize_features(self):
        """Normalize features using StandardScaler"""
        print("\nNormalizing features...")
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("Features normalized")
    
    def train_decision_tree(self):
        """Train Decision Tree model"""
        print("\n" + "="*60)
        print("TRAINING DECISION TREE MODEL")
        print("="*60)
        
        self.dt_model = DecisionTreeClassifier(
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE
        )
        
        self.dt_model.fit(self.X_train, self.y_train)
        print("Decision Tree training complete")
        
        # Evaluate
        y_pred = self.dt_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\nDecision Tree Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(
    self.y_test, y_pred,
    labels=[0, 1, 2, 3, 4],
    target_names=[WARNING_LEVELS[i] for i in range(5)],
    zero_division=0  # Avoids division by zero warnings if a class is missing
))

        
        return accuracy
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        self.rf_model.fit(self.X_train, self.y_train)
        print("Random Forest training complete")
        
        # Evaluate
        y_pred = self.rf_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\nRandom Forest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(
    self.y_test, y_pred,
    labels=[0, 1, 2, 3, 4],
    target_names=[WARNING_LEVELS[i] for i in range(5)],
    zero_division=0  # Avoids division by zero warnings if a class is missing
))

        
        return accuracy
    
    def save_models(self):
        """Save trained models and scaler"""
        print("\nSaving models...")
        
        # Save Decision Tree
        joblib.dump(self.dt_model, os.path.join(MODEL_DIR, "decision_tree_model.pkl"))
        print("Decision Tree saved")
        
        # Save Random Forest
        joblib.dump(self.rf_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
        print("Random Forest saved")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        print("Scaler saved")
        
        # Save scaler parameters for Arduino
        scaler_params = {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'feature_names': FEATURE_COLUMNS
        }
        
        import json
        with open(os.path.join(MODEL_DIR, "scaler_params.json"), 'w') as f:
            json.dump(scaler_params, f, indent=2)
        print("Scaler parameters saved for Arduino")

def main():
    """Main training pipeline"""
    print("="*60)
    print("SMART FIRE DETECTION - MODEL TRAINING")
    print("="*60)
    
    trainer = FireDetectionTrainer()
    
    # Load data
    df = trainer.load_processed_data()
    if df is None:
        return
    
    # Prepare features
    X, y = trainer.prepare_features(df)
    
    # Split data
    trainer.split_data(X, y)
    
    # Normalize
    trainer.normalize_features()
    
    # Train models
    dt_accuracy = trainer.train_decision_tree()
    rf_accuracy = trainer.train_random_forest()
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nDecision Tree Accuracy: {dt_accuracy*100:.2f}%")
    print(f"Random Forest Accuracy: {rf_accuracy*100:.2f}%")
    print(f"\nModels saved to: {MODEL_DIR}")
    print("\nNext step: Run 'python src/export_model.py' to convert for Arduino")

if __name__ == "__main__":
    main()
