import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "BreastCancerModel"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
if len(versions) < 3:
    raise SystemExit("Need at least 3 registered versions. Run: python src\\train.py")

versions.sort(key=lambda v: int(v.version))
print("Registered versions:")
for v in versions:
    print("Version", v.version, "| run_id:", v.run_id, "| status:", v.status)

# Select the registered version whose source run has the highest F1.
best = None
best_f1 = -1.0
for v in versions:
    run = client.get_run(v.run_id)
    f1 = run.data.metrics.get("f1", -1.0)
    print("Version", v.version, "F1 =", f1)
    if f1 > best_f1:
        best_f1 = f1
        best = v

if best is None:
    raise SystemExit("Could not identify a best version.")

# Current MLflow registry workflow: aliases instead of legacy stages.
client.set_registered_model_alias(MODEL_NAME, "production", str(best.version))
client.set_registered_model_alias(MODEL_NAME, "candidate", str(best.version))

# Mark an older, non-production version as retired when available.
older = [v for v in versions if v.version != best.version]
if older:
    retired = older[0]
    client.set_model_version_tag(MODEL_NAME, str(retired.version), "lifecycle", "retired")
    client.update_model_version(
        name=MODEL_NAME,
        version=str(retired.version),
        description="Retired for this lab demonstration; retained for audit and rollback."
    )

client.update_model_version(
    name=MODEL_NAME,
    version=str(best.version),
    description=f"Selected for production alias because it has the highest held-out F1 ({best_f1:.4f})."
)
print("\nProduction alias -> version", best.version, "| F1 =", round(best_f1, 4))
print("Candidate alias -> version", best.version)
if older:
    print("Retired tag added to version", older[0].version)
print("Acceptance rule: choose the version with the highest F1 on the same held-out test split.")
