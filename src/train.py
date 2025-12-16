import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def eval_metrics(actual, pred, pred_proba):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    roc_auc = roc_auc_score(actual, pred_proba)
    return accuracy, precision, recall, f1, roc_auc

def train_model():
    # 1. Load Data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_data.csv')
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}.")
        
    df = pd.read_csv(DATA_PATH)
    
    # 2. Prepare Data
    # Drop ID and String columns automatically
    non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime', 'is_high_risk']
    drop_cols = list(set(id_cols + non_numeric))
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Setup MLflow
    mlflow.set_experiment("Credit_Risk_Hyperparameter_Tuning")
    
    with mlflow.start_run(run_name="RandomForest_Tuned"):
        logging.info("Starting Hyperparameter Tuning for Random Forest...")
        
        # Create a Pipeline (Scaling + Model)
        # This addresses the 'sklearn Pipeline' feedback
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42))
        ])
        
        # Define Hyperparameters to search
        param_grid = {
            'rf__n_estimators': [50, 100],
            'rf__max_depth': [5, 10, None],
            'rf__min_samples_split': [2, 5]
        }
        
        # Run Grid Search
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logging.info(f"Best Parameters found: {best_params}")
        
        # Predict
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        acc, prec, rec, f1, auc = eval_metrics(y_test, y_pred, y_proba)
        
        # Log Metrics & Params
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        # Log the Pipeline Model (Scaling included!)
        mlflow.sklearn.log_model(best_model, "model")
        
        logging.info(f"Final Model Metrics - Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    train_model()