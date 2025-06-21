from flask import Flask, jsonify
from flasgger import Swagger
from app.routes import api
import os
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import start_http_server

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "..", "templates")

app = Flask(
    __name__,
    static_folder=os.path.abspath(os.path.join(BASE_DIR, "..", "static")),
    template_folder=os.path.abspath(os.path.join(BASE_DIR, "..", "templates"))
)

Swagger(app)  # <- Active Swagger UI

# Exposer les métriques Prometheus (sur les routes Flask)
metrics = PrometheusMetrics(app)

# Optionnel : définir des labels pour l'endpoint
metrics.info('app_info', 'Flask StackOverflow Tagger API', version='1.0.0')

# Enregistrer les routes
app.register_blueprint(api)


# Gestion des erreurs globales
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Ressource non trouvée'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Erreur interne'}), 500


if __name__ == '__main__':
    # Expose les métriques sur http://localhost:8000/metrics
    start_http_server(8999)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
