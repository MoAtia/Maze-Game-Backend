from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

model_inference_latency = Histogram(
    "model_inference_latency_seconds", "Time taken for model inference"
)

input_feature_distribution = Histogram(
    "input_feature_distribution",
    "Histogram of average input feature values"
)

def setup_metrics(app):
    Instrumentator().instrument(app).expose(app)
