from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd 
import joblib
import uvicorn
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Iris Flower Classification API",
    description="API for predicting iris flower species using ML model",
    version="1.0.0"
)

# Global variable to store the loaded model
model = None

# Pydantic model for request validation
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", example=6.5)
    sepal_width: float = Field(..., description="Sepal width in cm", example=3.0)
    petal_length: float = Field(..., description="Petal length in cm", example=5.8)
    petal_width: float = Field(..., description="Petal width in cm", example=2.2)

# Pydantic model for response
class PredictionResponse(BaseModel):
    prediction: int
    species: str

# Map prediction to species name
SPECIES_MAP = {
    0: "setosa",
    1: "versicolor", 
    2: "virginica"
}

@app.on_event("startup")
async def load_model():
    """Load the ML model on application startup"""
    global model
    try:
        model = joblib.load('iris_mlp_model.joblib')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Iris Flower Classification API",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_flower(features: IrisFeatures):
    """
    Predict iris flower species based on flower measurements
    
    Parameters:
    - sepal_length: Length of sepal in cm
    - sepal_width: Width of sepal in cm  
    - petal_length: Length of petal in cm
    - petal_width: Width of petal in cm
    
    Returns:
    - prediction: Numeric class (0, 1, or 2)
    - species: Species name (setosa, versicolor, or virginica)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create DataFrame with proper column names
        test_point = pd.DataFrame([{
            "sepal length (cm)": features.sepal_length,
            "sepal width (cm)": features.sepal_width,
            "petal length (cm)": features.petal_length,
            "petal width (cm)": features.petal_width
        }])
        
        # Make prediction
        y_pred = model.predict(test_point)
        prediction = int(y_pred[0])
        
        # Map to species name
        species = SPECIES_MAP.get(prediction, "unknown")
        
        return PredictionResponse(
            prediction=prediction,
            species=species
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5136)