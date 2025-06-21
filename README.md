# StackOverflow Tagger API

Une API Flask déployée avec Docker pour suggérer automatiquement des tags StackOverflow à partir du titre et du corps d'une question.

## 🚀 Fonctionnalités principales

- Prédiction de tags via modèle ML (régression logistique préentraînée)
- Documentation Swagger via Flasgger
- Monitoring API avec Prometheus + Grafana
- Analyse du drift (Prometheus metrics personnalisées)
- Déploiement complet via Docker Compose
- Tracking et packaging avec MLflow Recipes

## 📁 Arborescence du projet

```
stackoverflow-tagger-api/
├── app/                # Code source Flask
├── static/             # Modèle ML entraîné (reg_logit_nlp.pkl)
├── templates/          # Interfaces HTML (Sneat)
├── mlflow_recipes/     # Pipeline MLflow
├── monitoring/         # Prometheus config
├── Dockerfile          # Conteneurisation API
├── docker-compose.yml  # Stack complet (API + MLflow + Prometheus + MinIO)
├── MLproject           # Entry point MLflow CLI
```

## 📦 Lancer le projet

```bash
docker-compose up --build
```

- API : [http://localhost:5001](http://localhost:5001)
- Swagger UI : [http://localhost:5001/apidocs](http://localhost:5001/apidocs)
- MLflow UI : [http://localhost:5002](http://localhost:5002)
- Grafana : [http://localhost:3000](http://localhost:3000)
- Prometheus : [http://localhost:9090](http://localhost:9090)

## 🔬 MLflow Recipes

```bash
mlflow run . -e evaluate -P input_path=data/test.csv -P output_path=predictions.csv
```

## 📎 Dépôt GitHub

👉 [Lien vers le dépôt GitHub](https://github.com/roro-17/stackoverflow-tagger-api.git)

# stackoverflow-tagger-api
