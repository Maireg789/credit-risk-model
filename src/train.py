import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set tracking URI to local database to ensure API can find it later
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'mlflow.db')
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

def eval_metrics(actual, pred, pred_proba):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    roc_auc = roc_auc_score(actual, pred_proba)
    return accuracy, precision, recall, f1, roc_auc

def train_model():
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_data.csv')
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}.")
        
    df = pd.read_csv(DATA_PATH)
    
    # Prepare Data
    non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime', 'is_high_risk']
    drop_cols = list(set(id_cols + non_numeric))
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Credit_Risk_Final_Comparison")
    
    # --- Model 1: Logistic Regression (Baseline) ---
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))])
        lr_pipe.fit(X_train, y_train)
        
        y_pred = lr_pipe.predict(X_test)
        y_proba = lr_pipe.predict_proba(X_test)[:, 1]
        
        acc, prec, rec, f1, auc = eval_metrics(y_test, y_pred, y_proba)
        
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(lr_pipe, "model")
        logging.info(f"Logistic Regression AUC: {auc:.4f}")

    # --- Model 2: Random Forest (Tuned) ---
    with mlflow.start_run(run_name="RandomForest_Tuned") as run:
        rf_pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=42))])
        
        param_grid = {
            'rf__n_estimators': [50, 100],
            'rf__max_depth': [5, 10]
        }
        
        grid = GridSearchCV(rf_pipe, param_grid, cv=3, scoring='roc_auc')
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        acc, prec, rec, f1, auc = eval_metrics(y_test, y_pred, y_proba)
        
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("roc_auc", auc)
        
        # REGISTER the best model to the MLflow Registry
        mlflow.sklearn.log_model(
            best_model, 
            "model", 
            registered_model_name="Credit_Risk_Model_Prod"
        )
        logging.info(f"Random Forest AUC: {auc:.4f}. Model registered as 'Credit_Risk_Model_Prod'.")

if __name__ == "__main__":
    train_model()