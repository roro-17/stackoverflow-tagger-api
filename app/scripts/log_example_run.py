
import mlflow
import time
import os

# -- Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
EXPERIMENT_NAME = "stackoverflow_tags_prediction"

# -- Connexion à MLflow
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    mlflow.log_param("example_param", 1)
    mlflow.log_metric("avg_precision_macro", 0.87)
    mlflow.log_metric("avg_precision_micro", 0.91)
    mlflow.log_metric("roc_auc_macro", 0.93)
    mlflow.log_metric("roc_auc_micro", 0.95)
    mlflow.log_metric("inference_time", 0.42)

    print("✅ Run loggé avec succès dans MLflow")
    time.sleep(1)