from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging
from typing import Literal
# from monitoring import instrumentator
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter, Histogram

from pathlib import Path
import time
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Importing libraries done.")

# Initialize FastAPI app
app = FastAPI()

# Allow all origins (for testing/dev only)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace with actual domain in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://moatia.github.io",
        "https://mlops-final-project-production-ed5b.up.railway.app"
        "http://localhost:5500",  # If you have a local dev frontend
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

num_cols = []
for i in range(1, 22):
    num_cols.append(f"x{i}")
    num_cols.append(f"y{i}")
    num_cols.append(f"z{i}")


# Expected order of features
all_features = num_cols
BASE_DIR = Path(__file__).resolve().parent / "models"
# Load model and transformer
try:
    with open(BASE_DIR/"xgboost.pkl", "rb") as f:
        model = joblib.load(f)
    logging.info("Model loaded successfully.")

    with open(BASE_DIR/"label_encoder.pkl", "rb") as f:
        label_encoder = joblib.load(f)
    logging.info("Label Encoder loaded successfully.")

except Exception as e:
    logging.error(f"Error loading model or transformer: {e}")
    raise


# --- Model-related metric ---
model_inference_latency = Histogram(
    "model_inference_latency_seconds", "Time taken for model inference"
)

# --- Data-related metric ---
input_feature_distribution = Histogram(
    "input_feature_distribution",
    "Histogram of average input feature values"
)

# Creates a metrics endpoint at /metrics
Instrumentator().instrument(app).expose(app)
logging.info("Metrics endpoint created.")


class Landmarks(BaseModel):

    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    x3: float
    y3: float
    z3: float
    x4: float
    y4: float
    z4: float
    x5: float
    y5: float
    z5: float
    x6: float
    y6: float
    z6: float
    x7: float
    y7: float
    z7: float
    x8: float
    y8: float
    z8: float
    x9: float
    y9: float
    z9: float
    x10: float
    y10: float
    z10: float
    x11: float
    y11: float
    z11: float
    x12: float
    y12: float
    z12: float
    x13: float
    y13: float
    z13: float
    x14: float
    y14: float
    z14: float
    x15: float
    y15: float
    z15: float
    x16: float
    y16: float
    z16: float
    x17: float
    y17: float
    z17: float
    x18: float
    y18: float
    z18: float
    x19: float
    y19: float
    z19: float
    x20: float
    y20: float
    z20: float
    x21: float
    y21: float
    z21: float


# Endpoints
@app.get("/")
def home():
    logging.info("Home endpoint accessed.")
    return {"message": "Welcome to the Hand gesture to directions API!"}

@app.get("/health")
def health():
    logging.info("Health check accessed.")
    return {"status": "ok"}

def preprocess_landmarks(landmarks):
    coords = np.array(landmarks).reshape(-1, 3)
    wrist = coords[0][:2]
    mid_tip = coords[12][:2]
    scale = np.linalg.norm(mid_tip - wrist)
    coords[:, :2] = (coords[:, :2] - wrist) / (scale + 1e-6)
    return coords.flatten()

after_processing_features = []
for i in range(2, 22):
    after_processing_features.append(f"x{i}")
    after_processing_features.append(f"y{i}")

@app.post("/predict")
def predict(data: Landmarks):
    logging.info(f"Received data for prediction: {data.model_dump()}")

    try:
        # Create input in correct order
        
        input_data = [[getattr(data, col) for col in all_features]]
        logging.info(f"Ordered input data: {input_data}")

        input_feature_distribution.observe(data.x11)

        input_data = pd.DataFrame(input_data, columns=all_features)

        # preprocess input data
        processed = preprocess_landmarks(input_data)
        # Remove Zs components
        processed = processed.reshape(-1, 3)[:, :2].flatten()
        # Remove wrist point
        processed = processed[2:].reshape(1, -1)


        input_data = pd.DataFrame(processed, columns=after_processing_features)


        
        # Transform input
        # transformed = transformer.transform(input_data)
        logging.info("Data transformed successfully.")

        # Predict
        start_time = time.time()
        prediction = model.predict(input_data)
        gesture_name = label_encoder.inverse_transform(prediction)[0]
        duration = time.time() - start_time
        model_inference_latency.observe(duration)
        # prob = model.predict_proba(input_data)[0][1]
        logging.info(f"Prediction: {prediction[0]}, Probability: {gesture_name}")

        headers = {
            "prediction": prediction,
            "gesture_name": gesture_name
        }

        return {
            "prediction": int(prediction[0]),
            "gesture_name": gesture_name
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Change 8001 to your desired port
