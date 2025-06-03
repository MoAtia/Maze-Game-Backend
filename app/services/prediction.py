import joblib
import pandas as pd
from config import MODEL_PATH, ENCODER_PATH
from services.preprocessing import preprocess_landmarks
import logging
import time
from services.metrics import model_inference_latency, input_feature_distribution

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    logging.error(f"Failed loading model or encoder: {e}")
    raise

ALL_FEATURES = [f"{axis}{i}" for i in range(1, 22) for axis in ['x', 'y', 'z']]
PROCESSED_FEATURES = [f"{axis}{i}" for i in range(2, 22) for axis in ['x', 'y']]

def make_prediction(data):
    input_data = [[getattr(data, col) for col in ALL_FEATURES]]
    input_feature_distribution.observe(data.x11)
    df = pd.DataFrame(input_data, columns=ALL_FEATURES)
    processed = preprocess_landmarks(df)
    df_processed = pd.DataFrame(processed, columns=PROCESSED_FEATURES)

    start = time.time()
    prediction = model.predict(df_processed)
    duration = time.time() - start
    model_inference_latency.observe(duration)
    
    gesture = label_encoder.inverse_transform(prediction)[0]
    return int(prediction[0]), gesture
