# Citi Bike Prediction (Production)

Live app: https://citibikeprediction-app-ajggpbwskcodhpvyiumn7o.streamlit.app/

## Overview
This project predicts near-term Citi Bike trip demand by station in Jersey City and serves results through a Streamlit dashboard.

## Architecture
- Data source: Citi Bike trip history (scheduled ingestion)
- Feature Store: Hopsworks
  - `time_series_hourly_feature_group_citi_bike` (v1)
  - `time_series_hourly_feature_view_citi_bike` (v1)
- Model Registry: Hopsworks
  - `model_demand_predictor_next6hours`
- Prediction store:
  - `bike_6hours_model_prediction_citibike` (v1)
- Frontend: Streamlit (`frontend/frontend_v2.py`)

## Pipelines
GitHub Actions workflows run on schedule and manual trigger:
- `citibike_rides_hourly_features_pipeline`
- `citibike_rides_hourly_inference_pipeline`
- `citibike_rides_model_training_pipeline`

## Run Locally
1. Create env and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Add environment variables:
   - `HOPSWORKS_PROJECT_NAME`
   - `HOPSWORKS_API_KEY`
   - `MLFLOW_TRACKING_URI`
   - `MLFLOW_TRACKING_USERNAME`
   - `MLFLOW_TRACKING_PASSWORD`
3. Start app:
   - `streamlit run frontend/frontend_v2.py`

## Deployment
- Platform: Streamlit Community Cloud
- Main file: `frontend/frontend_v2.py`
- Python: `3.11` (pinned via `runtime.txt` and `.python-version`)

## Notes
- If predictions are temporarily unavailable for the next hour, the app now shows a graceful message instead of crashing.
- Hopsworks client is pinned for backend compatibility.
