from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import os
from .pydantic_models import CreditScoringRequest, CreditScoringResponse

app = FastAPI(title="Credit Risk Scoring API")

# Load Model and Features at startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'api', 'model_data.pkl')

model = None
required_features = []

@app.on_event("startup")
def load_model():
    global model, required_features
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            model = data["model"]
            required_features = data["features"]
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/")
def home():
    return {"message": "Credit Risk Scoring API is Online"}

@app.post("/predict", response_model=CreditScoringResponse)
def predict(request: CreditScoringRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input dict to DataFrame
        input_data = request.features
        df_input = pd.DataFrame([input_data])
        
        # Align columns with training data
        # This ensures missing columns are filled with 0 (e.g., missing One-Hot columns)
        df_input = df_input.reindex(columns=required_features, fill_value=0)
        
        # Predict
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        
        return {
            "is_high_risk": int(prediction),
            "risk_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))