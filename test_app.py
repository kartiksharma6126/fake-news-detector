"""
test_app.py
===========
A small pytest suite for the Flask API.

Run from the project root:
    pytest -v

NOTE: importing app.py loads the model once. If no saved model exists yet, it
auto-trains on the built-in sample dataset (takes a second), so the tests run
even on a fresh clone with no data/ files.
"""

import os
import sys

# Make sure imports work no matter where pytest is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

import app as app_module  # noqa: E402  (importing trains/loads the model once)

# A realistic-looking paragraph with comfortably more than 20 words.
VALID_TEXT = (
    "City officials announced on Tuesday that the downtown bridge repair project "
    "will begin next month and is expected to last through the fall. The council "
    "approved the funding last spring after engineers reported structural wear, "
    "and drivers will be redirected to two nearby crossings during construction."
)


@pytest.fixture(scope="module")
def client():
    """A Flask test client; the model is already loaded at import time."""
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Verification Desk" in response.data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model"] == "loaded"


def test_predict_valid_text(client):
    """A valid request returns a label, a confidence between 0 and 100, and top words."""
    response = client.post("/predict", json={"text": VALID_TEXT})
    assert response.status_code == 200
    body = response.get_json()
    assert body["label"] in ("FAKE", "REAL")
    assert 0 <= body["confidence"] <= 100
    assert isinstance(body["top_words"], list)
    assert body["words_analyzed"] >= 20


def test_predict_empty_text(client):
    """Empty input is rejected with a friendly 400 error."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_missing_text_key(client):
    """A JSON body without the 'text' key is also rejected."""
    response = client.post("/predict", json={"something": "else"})
    assert response.status_code == 400


def test_predict_too_short(client):
    """Fewer than 20 words is rejected with a helpful message."""
    response = client.post("/predict", json={"text": "This headline is far too short."})
    assert response.status_code == 400
    assert "20 words" in response.get_json()["error"]


def test_predict_too_long(client):
    """Over 20,000 characters is rejected."""
    long_text = "word " * 8000  # 40,000 characters -> over the 20,000 limit
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 400
    assert "too long" in response.get_json()["error"]


def test_predict_not_json(client):
    """A non-JSON body is rejected with a 400."""
    response = client.post("/predict", data="plain text, not json")
    assert response.status_code == 400
