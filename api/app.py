from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from .schemas import PropertyData
import os

app = FastAPI(title="Housing Price Prediction API")

# Global variables to hold models
models = {}

@app.on_event("startup")
def load_models():
    """Load models on startup to avoid latency per request"""
    # Adjust paths if your hierarchy is different
    sale_path = "models/model_sale.pkl"
    rent_path = "models/model_rent.pkl"
    
    if os.path.exists(sale_path):
        models["sale"] = joblib.load(sale_path)
        print("Sale model loaded.")
    
    if os.path.exists(rent_path):
        models["rent"] = joblib.load(rent_path)
        print("Rent model loaded.")

@app.get("/")
def index():
    return {"message": "Housing Price Prediction API is running. Use /predict/sale or /predict/rent"}

@app.post("/predict/{mode}")
def predict(mode: str, data: PropertyData):
    """
    Predicts price based on mode (sale/rent) and input data.
    """
    if mode not in ["sale", "rent"]:
        raise HTTPException(status_code=400, detail="Mode must be 'sale' or 'rent'")
    
    if mode not in models:
        raise HTTPException(status_code=500, detail=f"Model for {mode} not loaded")
    
    # Convert Pydantic object to Pandas DataFrame
    # This is crucial because Scikit-learn Pipeline expects a DataFrame with column names
    input_data = data.dict()
    df = pd.DataFrame([input_data])
    
    try:
        model = models[mode]
        # The pipeline handles scaling and one-hot encoding automatically!
        prediction = model.predict(df)
        return {
            "mode": mode,
            "predicted_price": float(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))