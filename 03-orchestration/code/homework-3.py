#!/usr/bin/env python
# homework.py

import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from prefect import flow, task

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-homework")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


@task(retries=3, retry_delay_seconds=10)
def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    print(f"Loaded {year}-{month:02d}: {len(df)} rows")
    return df


@task
def create_features(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task
def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        params = {
            'learning_rate': 0.1,
            'max_depth': 10,
            'min_child_weight': 1.0,
            'objective': 'reg:squarederror',
            'seed': 42
        }

        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=20
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        print(f"RMSE: {rmse:.4f}")

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


@flow(name="nyc-taxi-homework")
def main_flow():
    # March 2023 = train, April 2023 = validation
    df_train = read_dataframe(2023, 3)
    df_val = read_dataframe(2023, 4)

    X_train, dv = create_features(df_train)
    X_val, _ = create_features(df_val, dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")


if __name__ == "__main__":
    main_flow()