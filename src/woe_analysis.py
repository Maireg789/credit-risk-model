import pandas as pd
import numpy as np
import os

def calculate_iv(df, feature, target):
    """Calculates Information Value (IV) for a feature"""
    lst = []
    df[feature] = df[feature].fillna("Missing")
    
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': df[df[feature] == val].count()[feature],
            'Good': df[(df[feature] == val) & (df[target] == 0)].count()[feature],
            'Bad': df[(df[feature] == val) & (df[target] == 1)].count()[feature]
        })

    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    
    return dset['IV'].sum()

if __name__ == "__main__":
    # Load processed data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_data.csv')
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print("--- Information Value (IV) Analysis ---")
        
        # Check IV for a few categorical columns
        for col in ['ProductCategory', 'ChannelId', 'PricingStrategy']:
            if col in df.columns:
                iv = calculate_iv(df, col, 'is_high_risk')
                print(f"Feature: {col} | IV: {iv:.4f}")
                if iv > 0.3:
                    print(f"  -> {col} is a strong predictor.")
                elif iv < 0.02:
                    print(f"  -> {col} is a weak predictor.")
    else:
        print("Run feature_engineering.py first!")