def train_model():
    # 1. Load Data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_data.csv')
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}. Run feature_engineering.py first.")
        
    df = pd.read_csv(DATA_PATH)
    
    # 2. Prepare X (Features) and y (Target)
    # Identify non-numeric columns automatically
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Standard ID columns to drop
    id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime', 'is_high_risk']
    
    # Combine lists to drop
    drop_cols = list(set(id_cols + non_numeric_cols))
    
    # Ensure we only drop columns that actually exist
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    logging.info(f"Dropping these columns before training: {drop_cols}")
    
    X = df.drop(columns=drop_cols)
    y = df['is_high_risk']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Setup MLflow Experiment
    mlflow.set_experiment("Credit_Risk_Model_Experiment")
    
    # --- Model A: Logistic Regression ---
    with mlflow.start_run(run_name="Logistic_Regression"):
        logging.info("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        
        # Predict
        y_pred = lr.predict(X_test)
        y_proba = lr.predict_proba(X_test)[:, 1]
        
        # Evaluate
        acc, prec, rec, f1, auc = eval_metrics(y_test, y_pred, y_proba)
        
        # Log params and metrics
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)
        
        # Register Model
        mlflow.sklearn.log_model(lr, "model")
        logging.info(f"Logistic Regression - Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # --- Model B: Random Forest ---
    with mlflow.start_run(run_name="Random_Forest"):
        logging.info("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        
        # Evaluate
        acc, prec, rec, f1, auc = eval_metrics(y_test, y_pred, y_proba)
        
        # Log params and metrics
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)
        
        # Register Model
        mlflow.sklearn.log_model(rf, "model")
        logging.info(f"Random Forest - Accuracy: {acc:.4f}, AUC: {auc:.4f}")