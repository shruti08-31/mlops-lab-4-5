# MLOps Lab 4–5: Advanced MLflow and DVC

This lab demonstrates experiment tracking, model signatures, artifacts, three registered model versions, lifecycle aliases/tags, DVC dataset versioning, and Git/DVC/MLflow integration.

## Dataset
The dataset is scikit-learn's Breast Cancer Wisconsin dataset, exported to `data/dataset.csv` by `src/prepare_data.py`. It contains 569 rows, 30 feature columns, and one target column.

## Run
Activate `.venv`, install `requirements.txt`, then run:
1. `python src\prepare_data.py`
2. `dvc init`
3. `dvc add data\dataset.csv`
4. `git add .`
5. `git commit -m "Track initial project and dataset"`
6. `python src\train.py`
7. `python src\lifecycle.py`
8. `mlflow ui --backend-store-uri sqlite:///mlflow.db`

Open http://127.0.0.1:5000 in a browser.

## Reproducibility
Git tracks source code and DVC metadata. DVC stores dataset content in its cache/remote. MLflow stores run parameters, metrics, artifacts, model signatures, and registered model versions.
