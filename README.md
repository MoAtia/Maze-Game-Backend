# Maze Game Backend

## Overview
This repository provides an API for Maze game application, designed to be integrated into interactive applications such as games. The core of the system is a machine learning model served using FastAPI, enabling efficient and scalable inference.


## Features
- **Model Serving:** The hand gesture classification model is served via a FastAPI application for low-latency predictions.
- **Unit Testing:** Initial unit tests are provided (see `tests/test_api.py`) to ensure API reliability and correct error handling.
- **Containerization:** The application is containerized using Docker for consistent deployment across environments.
- **Docker Compose:** A `docker-compose.yml` file is provided for orchestrating the API, monitoring stack, and supporting services with a single command.

- **Monitoring & Observability:**
  - **Prometheus Metrics Exposure:** The API exposes Prometheus-compatible metrics at `/metrics` for real-time monitoring of model, data, and server health.
  - **Grafana Dashboard:** A sample `dashboard.json` is included for visualizing key metrics in Grafana.
  - **Three Key Metrics:**
    - **Model Metric:** The letancy of prediction, to ensure the real-time performance and avoid any bottelneck by taking consedring the latency Vs model complexity tradoff.
    - **Data Metric:** Data drift monitoring by track the landmarks' coordinates distribution.
    - **Server Metric:** Request latency logging to ensure API performance, identify bottlenecks and to identify the proper rate for rendering the game.


## API Endpoints
- `GET /` — Health check and welcome message.
- `POST /predict` — Accepts a JSON payload with hand landmarks and returns the predicted gesture and confidence.
- `GET /metrics` — Exposes Prometheus metrics for monitoring.

## Monitoring & Observability
- **Prometheus Metrics:** Access at `http://localhost:8000/metrics` when running locally or via Docker Compose.
- **Grafana Dashboard:** Import the provided `dashboard.json` into Grafana to visualize model confidence, data drift, and latency metrics.
- **docker-compose.yml:** Orchestrates the API, Prometheus, and Grafana for a complete monitoring stack.