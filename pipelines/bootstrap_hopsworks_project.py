import os
import sys
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config as config
from src.data_utils import (
    fetch_batch_raw_data_full,
    transform_raw_data_into_ts_data,
    transform_ts_data_info_features,
    transform_ts_data_info_features_and_target,
)
from src.inference import get_feature_store, get_hopsworks_project
from src.pipeline_utils import get_pipeline


def main() -> None:
    print("Step 1/5: Fetching historical Citi Bike data...")
    current_date = pd.Timestamp.now(tz="America/New_York").ceil("h")
    fetch_data_from = current_date - timedelta(days=180)
    rides = fetch_batch_raw_data_full(fetch_data_from, current_date)
    ts_data = transform_raw_data_into_ts_data(rides, interval_hours=1)
    ts_data["pickup_hour"] = pd.to_datetime(ts_data["pickup_hour"], utc=True)
    print(f"Time-series rows: {len(ts_data)}")

    print("Step 2/5: Creating/upserting feature group...")
    fs = get_feature_store()
    feature_group = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        description="Hourly Citi Bike ride counts per station.",
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_hour",
    )
    feature_group.insert(
        ts_data, write_options={"wait_for_job": True, "operation": "upsert"}
    )

    print("Step 3/5: Creating/retrieving feature view...")
    try:
        fs.create_feature_view(
            name=config.FEATURE_VIEW_NAME,
            version=config.FEATURE_VIEW_VERSION,
            query=feature_group.select_all(),
            labels=[],
            description="Feature view over hourly Citi Bike station demand.",
        )
        print("Feature view created.")
    except Exception:
        fs.get_feature_view(
            name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
        )
        print("Feature view already exists.")

    print("Step 4/5: Training and registering initial model...")
    features, targets = transform_ts_data_info_features_and_target(
        ts_data, window_size=24 * 28, step_size=12
    )
    train_cutoff = features["pickup_hour"].quantile(0.8)
    train_mask = features["pickup_hour"] < train_cutoff
    X_train, y_train = features[train_mask].copy(), targets[train_mask]
    X_test, y_test = features[~train_mask].copy(), targets[~train_mask]

    pipeline = get_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, predictions))
    print(f"Model test MAE: {test_mae:.4f}")

    model_path = config.MODELS_DIR / "lgb_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    project = get_hopsworks_project()
    mr = project.get_model_registry()
    model = mr.sklearn.create_model(
        name=config.MODEL_NAME,
        metrics={"test_mae": test_mae},
        description="Initial LightGBM model for Citi Bike next-hour demand.",
        input_example=X_train.sample(min(20, len(X_train))),
        model_schema=model_schema,
    )
    model.save(str(model_path))
    print("Model registered.")

    print("Step 5/5: Building predictions feature group for frontend...")
    inference_features = transform_ts_data_info_features(
        ts_data, window_size=24 * 28, step_size=23
    )
    inference_predictions = pipeline.predict(inference_features)
    predictions_df = pd.DataFrame(
        {
            "pickup_location_id": inference_features["pickup_location_id"].values,
            "pickup_hour": pd.to_datetime(
                inference_features["pickup_hour"], utc=True
            ).values,
            "predicted_demand": pd.Series(inference_predictions).round(0).clip(lower=0).values,
        }
    )

    pred_fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTION,
        version=1,
        description="Predictions from model_demand_predictor_next6hours.",
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_hour",
    )
    pred_fg.insert(
        predictions_df, write_options={"wait_for_job": True, "operation": "upsert"}
    )
    print(f"Prediction rows upserted: {len(predictions_df)}")
    print("Bootstrap complete.")


if __name__ == "__main__":
    main()
