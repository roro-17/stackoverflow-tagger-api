#!/bin/sh

# Attendre que le serveur MinIO soit disponible
sleep 10

# Configurer mc pour MinIO
mc alias set local http://minio:9000 mlflowadmin mlflowsecret

# Créer le bucket "mlflow" si non existant
mc mb local/mlflow || true
