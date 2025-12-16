from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import mlflow.sklearn
import os

app = FastAPI(title="Credit Risk Scoring API")

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)

# Point to the database in the root folder
DB_PATH = os.path.join(root_dir, 'mlflow.db')
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

# Global variable to hold the model
model = None

# --- INPUT/OUTPUT MODELS ---
class CreditScoringRequest(BaseModel):
    features: Dict[str, Any]

class CreditScoringResponse(BaseModel):
    is_high_risk: int
    risk_probability: float

# --- STARTUP EVENT ---
@app.on_event("startup")
def load_production_model():
    global model
    model_name = "Credit_Risk_Model_Prod"
    
    print(f"Connecting to MLflow DB at: {DB_PATH}")
    
    try:
        model_uri = f"models:/{model_name}/Latest"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Successfully loaded '{model_name}' from MLflow Registry.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Tip: Ensure you ran 'python src/train.py' first.")

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"message": "Credit Risk Scoring API is Online"}

@app.post("/predict", response_model=CreditScoringResponse)
def predict(request: CreditScoringRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
    
    try:
        # 1. Convert input to DataFrame
        df_input = pd.DataFrame([request.features])
        
        # 2. Add missing columns with 0
        if hasattr(model, "feature_names_in_"):
            expected_cols = model.feature_names_in_
            df_input = df_input.reindex(columns=expected_cols, fill_value=0)
        
        # 3. Predict
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        
        return {
            "is_high_risk": int(prediction),
            "risk_probability": float(probability)
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction Error: {str(e)}")
    