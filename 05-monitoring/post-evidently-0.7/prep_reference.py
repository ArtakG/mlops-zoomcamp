"""One-shot prep: train baseline LinearRegression on Jan 2022 green taxi data,
then dump models/lin_reg.bin and data/reference.parquet so
evidently_metrics_calculation.py has what it needs."""
import os
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression

os.makedirs("models", exist_ok=True)

jan = pd.read_parquet("data/green_tripdata_2022-01.parquet")
jan["duration_min"] = (jan.lpep_dropoff_datetime - jan.lpep_pickup_datetime).dt.total_seconds() / 60
jan = jan[(jan.duration_min >= 0) & (jan.duration_min <= 60)]
jan = jan[(jan.passenger_count > 0) & (jan.passenger_count <= 8)]

num = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat = ["PULocationID", "DOLocationID"]

train, val = jan[:30000].copy(), jan[30000:].copy()

m = LinearRegression()
m.fit(train[num + cat], train["duration_min"])

val["prediction"] = m.predict(val[num + cat])

with open("models/lin_reg.bin", "wb") as f:
    dump(m, f)

val.to_parquet("data/reference.parquet")

print("OK - models/lin_reg.bin and data/reference.parquet written")
