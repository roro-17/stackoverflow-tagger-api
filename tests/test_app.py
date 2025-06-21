import pytest
from flask import Flask
from app.routes import api


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(api)
    app.config['TESTING'] = True
    return app.test_client()


def test_api_predict_valid(client):
    """
    Test the predict API endpoint with valid input.
    """
    response = client.post('/api/rest/v1/questions/tags', json={
        "title": "How to use Flask with MLflow?",
        "body": "I want to integrate model prediction with Flask using MLflow and Docker."
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "Predicted_Tags" in data
    assert isinstance(data["Predicted_Tags"], list)
    assert len(data["Predicted_Tags"]) > 0


def test_api_predict_missing_fields(client):
    """
    Test the predict API endpoint with missing fields.
    """
    response = client.post('/api/rest/v1/questions/tags', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_api_predict_invalid_content_type(client):
    """
    Test the predict API endpoint with invalid content type.
    """
    response = client.post('/api/rest/v1/questions/tags', data="just text", content_type="text/plain")
    assert response.status_code == 415
    data = response.get_json()
    assert "error" in data
