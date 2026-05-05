#!/usr/bin/env python
# homework-3.py

import gc
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
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
    columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime',
               'PULocationID', 'DOLocationID', 'trip_distance']
    df = pd.read_parquet(url, columns=columns)

    print(f"Raw records loaded: {len(df)}")

    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    print(f"After filtering: {len(df)}")

    # Keep PU and DO separate — no combination feature
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


@task
def create_features(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


@task
def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        print(f"Intercept: {lr.intercept_:.2f}")  # <-- Q5 answer

        y_pred = lr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        print(f"RMSE: {rmse:.4f}")

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")  # <-- Q6: check MLModel size

        return run.info.run_id


@flow(name="nyc-taxi-homework")
def main_flow():
    # Train
    df_train = read_dataframe(2023, 3)
    X_train, dv = create_features(df_train)
    y_train = df_train['duration'].values.copy()
    del df_train
    gc.collect()

    # Validation
    df_val = read_dataframe(2023, 4)
    X_val, _ = create_features(df_val, dv)
    y_val = df_val['duration'].values.copy()
    del df_val
    gc.collect()

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")


if __name__ == "__main__":
    main_flow()