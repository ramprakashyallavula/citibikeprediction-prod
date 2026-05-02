# Citi Bike Prediction (Production)

Live App: https://citibikeprediction-app-ajggpbwskcodhpvyiumn7o.streamlit.app/

The goal of this project is to take raw trip data, continuously transform it into model-ready features, generate predictions on a schedule, and serve those results in a live dashboard.

## What This Project Does
If you open the app, you can see predicted Citi Bike demand for Jersey City stations for the upcoming hour, along with a station-level trend view.



## How It Works (End-to-End)
Think of this project as a 4-step loop that keeps refreshing:

1. Collect and prepare ride data
- The feature pipeline pulls Citi Bike trip data.
- It cleans and validates the records.
- It aggregates rides into hourly station-level time series.
- It writes this data to Hopsworks Feature Store.

2. Build feature sets for learning and inference
- Historical windows are converted into lag-based features (for example, prior hourly ride counts).
- These engineered features are used both for model training and live prediction.

3. Train and register model
- The training pipeline fits a LightGBM demand model.
- The model is registered in Hopsworks Model Registry.
- Model versions and metrics are tracked for reproducibility.

4. Run inference and publish predictions
- The inference pipeline loads the latest model and recent feature windows.
- It predicts upcoming demand by station.
- Predictions are stored back in Hopsworks and read by Streamlit.

## Project Architecture
- Data Source: Citi Bike trip history
- Feature Store: Hopsworks
  - `time_series_hourly_feature_group_citi_bike` (v1)
  - `time_series_hourly_feature_view_citi_bike` (v1)
- Model Registry: Hopsworks
  - `model_demand_predictor_next6hours`
- Prediction Group:
  - `bike_6hours_model_prediction_citibike` (v1)
- App Layer: Streamlit (`frontend/frontend_v2.py`)

## Automation / Pipelines
These GitHub Actions workflows keep the system updated:
- `citibike_rides_hourly_features_pipeline`
- `citibike_rides_hourly_inference_pipeline`
- `citibike_rides_model_training_pipeline`

## Running Locally
1. Create environment and install dependencies
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

2. Set environment variables
- `HOPSWORKS_PROJECT_NAME`
- `HOPSWORKS_API_KEY`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

3. Start app
- `streamlit run frontend/frontend_v2.py`

## Deployment
- Platform: Streamlit Community Cloud
- Entry file: `frontend/frontend_v2.py`
- Python version: `3.11` (pinned via `runtime.txt` and `.python-version`)

## Reliability Notes
- The app handles temporary no-prediction windows gracefully.
- Hopsworks client is pinned to a backend-compatible version for stable execution.

