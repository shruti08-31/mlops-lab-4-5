import os
import subprocess
import time
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "MLOps-Lab-4-5"
REGISTERED_MODEL_NAME = "BreastCancerModel"

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Load dataset
df = pd.read_csv("data/dataset.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Three candidate models
candidates = [
    {
        "name": "candidate_1",
        "n_estimators": 50,
        "max_depth": 5
    },
    {
        "name": "candidate_2",
        "n_estimators": 100,
        "max_depth": None
    },
    {
        "name": "candidate_3",
        "n_estimators": 200,
        "max_depth": 10
    }
]

# Get Git commit ID
git_commit = "unknown"

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL,
        text=True
    ).strip()
except Exception:
    pass

# Get DVC dataset hash
dvc_hash = "unknown"

try:
    with open(
        "data/dataset.csv.dvc",
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            if "md5:" in line:
                dvc_hash = line.split("md5:", 1)[1].strip()
                break
except Exception:
    pass

results = []

# Train and log all three candidates
for cfg in candidates:

    model = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        random_state=42
    )

    with mlflow.start_run(
        run_name=cfg["name"]
    ) as run:

        start = time.time()

        model.fit(X_train, y_train)

        train_seconds = time.time() - start

        pred = model.predict(X_test)

        # Calculate evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, pred, zero_division=0
            ),
            "training_seconds": train_seconds
        }

        # Log parameters
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": cfg["n_estimators"],
            "max_depth": str(cfg["max_depth"]),
            "test_size": 0.2,
            "random_state": 42,
            "dataset_rows": len(df),
            "dataset_columns_including_target": len(df.columns),
            "git_commit": git_commit,
            "dvc_dataset_hash": dvc_hash
        })

        # Log metrics
        mlflow.log_metrics(metrics)

        # Artifact 1: Confusion matrix
        cm = confusion_matrix(y_test, pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["malignant", "benign"]
        )

        disp.plot()

        plt.title(
            cfg["name"] + " - Confusion Matrix"
        )

        fig_path = (
            "reports/figures/"
            + cfg["name"]
            + "_confusion_matrix.png"
        )

        plt.savefig(
            fig_path,
            bbox_inches="tight"
        )

        plt.close()

        mlflow.log_artifact(fig_path)

        # Artifact 2: Evaluation report
        report_path = (
            "reports/"
            + cfg["name"]
            + "_evaluation.txt"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("Run: " + cfg["name"] + "\n")
            f.write("Parameters: " + str(cfg) + "\n")

            for key, value in metrics.items():
                f.write(
                    f"{key}: {value:.6f}\n"
                )

        mlflow.log_artifact(report_path)

        # Model signature and example input
        signature = infer_signature(
            X_train,
            model.predict(X_train.head(5))
        )

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # Save results
        results.append({
            "run_name": cfg["name"],
            "run_id": run.info.run_id,
            **metrics
        })

        print("\nCandidate:", cfg["name"])
        print("Run ID:", run.info.run_id)
        print("Metrics:", metrics)


# Save comparison table
results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nSaved reports/model_comparison.csv")

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print(
    "\nRegistered model name:",
    REGISTERED_MODEL_NAME
)

print(
    "\nAll three training runs completed."
)