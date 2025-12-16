import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def save_model():
    # 1. Load Data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_data.csv')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'src', 'api', 'model_data.pkl')
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logging.info(f"Data loaded: {df.shape}")

    # 2. Prepare Features
    # Drop non-numeric and ID columns (Same logic as train.py)
    non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime', 'is_high_risk']
    drop_cols = list(set(id_cols + non_numeric))
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df['is_high_risk']
    
    # 3. Train Final Model (Random Forest)
    logging.info("Training final Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    
    # 4. Save Model AND Column Names
    # We need the column names to handle One-Hot Encoding alignment in the API
    model_data = {
        "model": rf,
        "features": X.columns.tolist()
    }
    
    # Create api directory if not exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(model_data, f)
        
    logging.info(f"Model and features saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    save_model()