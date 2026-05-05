# Homework 2 — Experiment Tracking with MLflow

## Q1. Install MLflow

Installed MLflow in the local Python environment and ran:

```bash
python -m mlflow --version
```

Output:

```
python -m mlflow, version 3.11.1
```

**Answer: 3.11.1**

![Q1 — MLflow version](Q1-mlflow-version.png)

---

## Q2. Download and preprocess the data

Then ran the preprocessing script:

```bash
python preprocess_data.py --raw_data_path ./input --dest_path ./output
```

Listing the resulting folder:

4
```

---

## Q3. Train a model with autolog

Modified `train.py` to enable MLflow autologging, set the tracking URI to the
local server, and wrapped training in `with mlflow.start_run():`.

Output:

```

RMSE: 5.4312
min_samples_split: 2
🏃 View run abrasive-deer-950 at:
   http://localhost:5000/#/experiments/2/runs/870bf25a42fc4cad968378777de8390b
```

The autologged `Parameters` section in the MLflow UI confirms the value.

**Answer: `min_samples_split = 2`**

![Q3 — train.py output](Q3-train-output.png)
![Q3 — MLflow run parameters](Q3-mlflow-params.png)

---

## Q4. Launch the tracking server locally

To run a local tracking server with both a SQLite metadata store and a local
artifact store, you must pass `default-artifact-root` in addition to
`backend-store-uri`:

```bash

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 127.0.0.1 \
  --port 5000
```

- `backend-store-uri` — where MLflow stores metadata (experiments, runs,
  params, metrics) → SQLite db (`mlflow.db`).
- `default-artifact-root` — where MLflow stores artifacts (models, files) →
  local `./artifacts` folder.

**Answer: `default-artifact-root`**

![Q4 — tracking server UI](Q4-tracking-server.png)

---

## Q5. Tune model hyperparameters

Modified the `objective` function in `hpo.py` to log each trial as its own
MLflow run, recording the hyperparameters and the validation RMSE. No
autologging — only the values needed for the question:

```python
def objective(params):
    with mlflow.start_run():
        mlflow.log_params(params)
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
    return {'loss': rmse, 'status': STATUS_OK}
```

Ran the optimization (15 trials, default):

```bash
cd cohorts/2025/02-experiment-tracking/homework
python hpo.py
```

Opened the MLflow UI and sorted the runs in the `random-forest-hyperopt`
experiment by `rmse` ascending. The best run was `polite-midge-921` with
**RMSE = 5.3354**.

**Answer: `5.335`**

![Q5 — best RMSE in MLflow UI](Q5-mlflow-best-rmse.png)

---

## Q6. Promote the best model to the model registry

Updated `register_model.py` to:

1. After re-training the top 5 hyperopt runs in the new
   `random-forest-best-models` experiment (with autologging on the test set),
   query that experiment with `MlflowClient.search_runs` ordered by
   `metrics.test_rmse ASC`, picking the first result.
2. Register that run's model with `mlflow.register_model` using the URI
   `runs:/<RUN_ID>/model` and the registered name `random-forest-best-model`.

Diff (the lines added at the end of `run_register_model`):

```python
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
best_run = client.search_runs(
    experiment_ids=experiment.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.test_rmse ASC"]
)[0]

best_run_id = best_run.info.run_id
best_test_rmse = best_run.data.metrics["test_rmse"]
print(f"Best run_id: {best_run_id}")
print(f"Best test RMSE: {best_test_rmse:.4f}")

model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri=model_uri, name="random-forest-best-model")
```

Ran:

```bash
cd cohorts/2025/02-experiment-tracking/homework
python register_model.py
```

Tail of the output:

```
🏃 View run unruly-smelt-321 at:
   http://127.0.0.1:5000/#/experiments/2/runs/0671852a772744cdb1cd30df65d60729
...
Best run_id: 0671852a772744cdb1cd30df65d60729
Best test RMSE: 5.5674
Created version '2' of model 'random-forest-best-model'.
```

Verified in the MLflow UI:
- Experiment `random-forest-best-models` → top run by `test_rmse` shows
  **5.5674** for `unruly-smelt-321`.
- Models → `random-forest-best-model` exists with a registered version.

**Answer: `5.567`**

![Q6 — register_model.py output](Q6-register-output.png)
![Q6 — best test RMSE in MLflow UI](Q6-mlflow-best-models.png)
![Q6 — registered model in Model Registry](Q6-model-registry.png)

