
import pickle
import pandas as pd
import mlflow
import os
from app.utils import sentence_cleaner
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score, roc_auc_score
import time
from datetime import datetime


def evaluate_model(input_path, output_path="predictions.csv"):
    # Charger les données transformées
    df = pd.read_csv(input_path)

    # Appliquer le prétraitement
    df["processed"] = df["title"].fillna("") + " " + df["body"].fillna("")
    df["processed"] = df["processed"].apply(sentence_cleaner)

    # Charger le modèle de régression logistique
    model_path = "static/models/reg_logit_nlp.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Supposer que le modèle est un pipeline incluant la vectorisation
    y_pred_proba = model.predict_proba(df["processed"])
    predictions = model.predict(df["processed"])

    # Sauvegarder les prédictions
    df["predicted_tags"] = predictions
    df.to_csv(output_path, index=False)

    return df, model, y_pred_proba


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="predictions.csv")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002"))
    mlflow.set_experiment("stackoverflow_tags_prediction")

    with mlflow.start_run():
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("input_file", args.input_path)
        mlflow.log_param("run_time", datetime.utcnow().isoformat())

        # Évaluation
        df, model, y_pred_proba = evaluate_model(args.input_path, args.output_path)

        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="stackoverflow-tagger-logit"
        )

        # Évaluation des métriques
        if "tags" in df.columns:
            df["tags"] = df["tags"].apply(eval) if df["tags"].dtype == object else df["tags"]
            mlb = MultiLabelBinarizer()
            y_true = mlb.fit_transform(df["tags"])

            start_time = time.time()
            inference_time = time.time() - start_time

            avg_precision_macro = average_precision_score(y_true, y_pred_proba, average='macro')
            avg_precision_micro = average_precision_score(y_true, y_pred_proba, average='micro')
            roc_auc_macro = roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovr')
            roc_auc_micro = roc_auc_score(y_true, y_pred_proba, average='micro', multi_class='ovr')

            mlflow.log_metric("avg_precision_macro", avg_precision_macro)
            mlflow.log_metric("avg_precision_micro", avg_precision_micro)
            mlflow.log_metric("roc_auc_macro", roc_auc_macro)
            mlflow.log_metric("roc_auc_micro", roc_auc_micro)
            mlflow.log_metric("inference_time", inference_time)

        mlflow.log_artifact(args.output_path)