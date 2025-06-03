from fastapi import APIRouter, HTTPException
from models.landmarks import Landmarks
from services.prediction import make_prediction
import logging

router = APIRouter()

@router.get("/")
def home():
    logging.info("Home endpoint accessed.")
    return {"message": "Welcome to the Hand gesture to directions API!"}

@router.get("/health")
def health():
    logging.info("Health check accessed.")
    return {"status": "ok"}

@router.post("/predict")
def predict(data: Landmarks):
    try:
        logging.info("Received data for prediction.")
        prediction, gesture = make_prediction(data)
        return {
            "prediction": prediction,
            "gesture_name": gesture
        }
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
