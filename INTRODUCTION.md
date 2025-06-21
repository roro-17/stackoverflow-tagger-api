# INTRODUCTION

## 🎯 Objectif

Ce projet expose une API de type StackOverflow qui permet, à partir d’une question (titre + description), de prédire les tags les plus pertinents.

L'API est conçue pour :
- être déployée en Docker
- inclure un modèle ML préentraîné
- exposer une documentation Swagger
- être monitorée avec Prometheus + Grafana
- suivre les dérives de données (via métriques personnalisées)

## 📁 Structure des dossiers

- `app/` : routes Flask, modèle, prétraitement
- `templates/` : frontend HTML avec Sneat
- `static/models/` : contient `reg_logit_nlp.pkl` (modèle ML)
- `monitoring/` : configuration Prometheus
- `mlflow_recipes/` : évaluation automatisée via MLflow Recipes
- `docker-compose.yml` : infrastructure complète
