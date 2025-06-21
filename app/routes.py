from flask import Blueprint, request, render_template, jsonify
from app.utils import sentence_cleaner
from app.model import predict_tags
from prometheus_client import Gauge

api = Blueprint('api', __name__)

# Définir des métriques personnalisées pour le drift
title_length_avg = Gauge('title_length_avg', 'Longueur moyenne des titres')
body_word_count_avg = Gauge('body_word_count_avg', 'Nombre moyen de mots dans body')


@api.route('/', methods=['GET'])
def formulaire():
    errors = {}
    return render_template("html/form.html", errors=errors)


@api.route('/questions/tags', methods=['POST', 'GET'])
def resultats_tags():
    title = request.form.get('title', '').strip()
    body = request.form.get('body', '').strip()

    # Erreurs par champ
    errors = {}

    if not title:
        errors['title'] = "Le titre est requis."
    if not body:
        errors['body'] = "La description est requise."

    if errors:
        return render_template("html/form.html", errors=errors, title=title, body=body)

    # Traitement ML
    doc = sentence_cleaner(title) + sentence_cleaner(body)
    tags_pred = predict_tags(doc)
    return render_template("html/result.html", body=body, title=title, tags_prediction=tags_pred)


@api.route('/api/rest/v1/questions/tags', methods=['POST'])
def predict_api():
    """
    Prédiction automatique de tags StackOverflow à partir d'une question
    ---
    tags:
      - Tags Prediction
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            title:
              type: string
              example: How to train a model in scikit-learn?
            body:
              type: string
              example: I have a dataset and want to train a model to classify categories.
    responses:
      200:
        description: Liste des tags prédits
        schema:
          type: object
          properties:
            tags:
              type: array
              items:
                type: string
              example: ["python", "scikit-learn", "classification"]
      400:
        description: Champs manquants
      415:
        description: Mauvais format
      500:
        description: Erreur serveur
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Le corps de la requête doit être au format JSON"}), 415

        data = request.get_json()
        title = data.get('title', '').strip()
        body = data.get('body', '').strip()

        if not title or not body:
            return jsonify({"error": "Le titre et la description sont requis"}), 400

        # Metriques drift
        title_length_avg.set(len(title))
        body_word_count_avg.set(len(body.split()))

        tokens = sentence_cleaner(title) + sentence_cleaner(body)
        predicted_tags = predict_tags(tokens)

        return jsonify({'Predicted_Tags': predicted_tags}), 200

    except Exception as e:
        return jsonify({"error": "Erreur interne du serveur", "details": str(e)}), 500