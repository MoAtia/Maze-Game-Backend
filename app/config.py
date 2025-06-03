from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = BASE_DIR / "xgboost.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

CORS_ORIGINS = [
    "https://moatia.github.io",
    "https://mlops-final-project-production-ed5b.up.railway.app",
    "http://localhost:5500",
    "http://localhost:8080"
]
