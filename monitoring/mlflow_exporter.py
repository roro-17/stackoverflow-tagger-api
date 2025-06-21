
from prometheus_client import start_http_server, Gauge, CollectorRegistry, generate_latest
from flask import Flask, Response
import mlflow
import os
import logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
registry = CollectorRegistry()

# Définition des métriques MLflow que l’on souhaite exporter
metrics = {
    "avg_precision_macro": Gauge("mlflow_avg_precision_macro", "Average precision macro", ["run_id"], registry=registry),
    "avg_precision_micro": Gauge("mlflow_avg_precision_micro", "Average precision micro", ["run_id"], registry=registry),
    "roc_auc_macro": Gauge("mlflow_roc_auc_macro", "ROC AUC macro", ["run_id"], registry=registry),
    "roc_auc_micro": Gauge("mlflow_roc_auc_micro", "ROC AUC micro", ["run_id"], registry=registry),
    "inference_time": Gauge("mlflow_inference_time", "Inference time", ["run_id"], registry=registry)
}

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-ui:5002")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def update_metrics():
    logging.info("🔄 Tentative de connexion à MLflow...")
    try:
        client = mlflow.tracking.MlflowClient()

        # Essayer d'abord avec le nom connu
        experiment = client.get_experiment_by_name("stackoverflow_tags_prediction")

        # Si pas trouvé, utiliser le premier existant
        if experiment is None:
            logging.warning("❗ Aucun experiment nommé 'stackoverflow_tags_prediction'. Recherche du premier existant...")
            experiment_list = client.search_experiments()
            if not experiment_list:
                logging.error("🚫 Aucun experiment du tout n'existe dans MLflow.")
                return
            experiment = experiment_list[0]
            logging.info(f"📌 Utilisation de l'expérience : {experiment.name} (id={experiment.experiment_id})")
        else:
            logging.info(f"✅ Expérience trouvée : {experiment.name} (id={experiment.experiment_id})")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=10,
            order_by=["attributes.start_time DESC"]
        )
        logging.info(f"📦 {len(runs)} runs trouvés pour l'expérience {experiment.name}")

        for run in runs:
            run_id = run.info.run_id
            for key, gauge in metrics.items():
                value = run.data.metrics.get(key)
                if value is not None:
                    logging.info(f"📊 {key} [run={run_id}] = {value}")
                    gauge.labels(run_id=run_id).set(value)

    except Exception as e:
        logging.exception("💥 Erreur lors de l'accès à MLflow : %s", e)


@app.route("/metrics")
def metrics_endpoint():
    update_metrics()
    return Response(generate_latest(registry), mimetype="text/plain")

if __name__ == "__main__":
    print("🚀 MLflow Prometheus Exporter started on http://localhost:8001/metrics")
    app.run(host="0.0.0.0", port=8001)
