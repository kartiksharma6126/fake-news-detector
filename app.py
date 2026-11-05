"""
app.py
======
Flask backend for the Fake News Detector.

Endpoints:
    GET  /          -> serves the single-page frontend (templates/index.html)
    GET  /health    -> tiny status check used by the frontend "model ready" badge
    POST /predict   -> accepts {"text": "..."} and returns
                       {"label": "FAKE|REAL", "confidence": 0-100, "top_words": [...]}

The model is loaded ONCE when the app starts (never per request).
If no saved model exists yet, one is trained automatically on first run.

Run locally:
    python app.py
"""

import os
import threading
import time

import joblib
from flask import Flask, jsonify, render_template, request

import train_model
from train_model import MODEL_PATH, VECTORIZER_PATH, clean_text, explain_prediction

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Hard cap on request size (well above the 20,000-character text limit).
# This is a first line of defense against oversized payloads.
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB

# Input rules (kept as constants so they are easy to change and test)
MIN_WORDS = 20
MAX_CHARS = 20_000

# Simple in-memory rate limiter: 30 predictions per minute per IP address.
# (A production app might use Redis or a library; this keeps zero dependencies.)
RATE_LIMIT_MAX_HITS = 30
RATE_LIMIT_WINDOW_SECONDS = 60
_hits_by_ip = {}
_hits_lock = threading.Lock()


def _is_rate_limited(ip):
    """Return True if this IP has made too many requests in the last minute."""
    if app.config.get("TESTING"):  # never throttle the pytest test suite
        return False
    now = time.time()
    with _hits_lock:
        recent = [t for t in _hits_by_ip.get(ip, []) if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(recent) >= RATE_LIMIT_MAX_HITS:
            _hits_by_ip[ip] = recent
            return True
        recent.append(now)
        _hits_by_ip[ip] = recent
        return False


# ---------------------------------------------------------------------------
# Load (or train) the model ONCE at startup
# ---------------------------------------------------------------------------
def _load_model():
    if os.path.isfile(MODEL_PATH) and os.path.isfile(VECTORIZER_PATH):
        print("[startup] Loading saved model from the model/ folder...")
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH), "saved model (model/)"
    print("[startup] No saved model found — training one automatically on first run...")
    model, vectorizer, source = train_model.train(save=True)
    return model, vectorizer, source


MODEL, VECTORIZER, DATASET_SOURCE = _load_model()

# CORS note: the frontend is served by this same Flask app, so every request is
# same-origin and CORS is NOT needed (no flask-cors dependency required).


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Serve the single-page frontend."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Status endpoint so the UI can show a 'model ready' badge."""
    return jsonify(
        {
            "status": "ok",
            "model": "loaded",
            "dataset": DATASET_SOURCE,
            "min_words": MIN_WORDS,
            "max_chars": MAX_CHARS,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Analyze pasted text and return a verdict with confidence + explanation."""

    # --- 1. Parse the JSON body -------------------------------------------
    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return jsonify({"error": 'Please send a JSON body like {"text": "..."}.'}), 400

    text = str(payload.get("text") or "").strip()

    # --- 2. Validate the input ---------------------------------------------
    if not text:
        return jsonify({"error": "The text is empty. Paste a headline or article first."}), 400

    if len(text) > MAX_CHARS:
        return (
            jsonify(
                {
                    "error": "That text is too long ({:,} characters). "
                    "Please keep it under {:,} characters.".format(len(text), MAX_CHARS)
                }
            ),
            400,
        )

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        return (
            jsonify(
                {
                    "error": "Please paste at least {} words so the model has enough "
                    "writing style to judge (you sent {}).".format(MIN_WORDS, word_count)
                }
            ),
            400,
        )

    # --- 3. Basic rate limiting ---------------------------------------------
    if _is_rate_limited(request.remote_addr or "unknown"):
        return jsonify({"error": "Too many requests — please wait a minute and try again."}), 429

    # --- 4. Predict -----------------------------------------------------------
    try:
        features = VECTORIZER.transform([clean_text(text)])

        # probabilities[0] = P(FAKE), probabilities[1] = P(REAL)
        probabilities = MODEL.predict_proba(features)[0]
        best_index = int(probabilities.argmax())
        predicted_class = int(MODEL.classes_[best_index])  # 0 = FAKE, 1 = REAL
        confidence = round(float(probabilities[best_index]) * 100, 1)

        label = "REAL" if predicted_class == 1 else "FAKE"
        top_words = explain_prediction(text, MODEL, VECTORIZER, top_n=8)

        return jsonify(
            {
                "label": label,
                "confidence": confidence,
                "top_words": top_words,
                "words_analyzed": word_count,
            }
        )
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Something went wrong while analyzing the text. Please try again."}), 500


# ---------------------------------------------------------------------------
# Friendly error handlers (JSON for the API, simple text elsewhere)
# ---------------------------------------------------------------------------
def _wants_api():
    return request.path.startswith(("/predict", "/health"))


@app.errorhandler(404)
def handle_404(err):
    if _wants_api():
        return jsonify({"error": "Endpoint not found."}), 404
    return "404 — page not found", 404


@app.errorhandler(405)
def handle_405(err):
    if _wants_api():
        return jsonify({"error": "Method not allowed. Use POST /predict with a JSON body."}), 405
    return "405 — method not allowed", 405


@app.errorhandler(413)
def handle_413(err):
    return jsonify({"error": "Request too large. Please keep your text under 20,000 characters."}), 413


@app.errorhandler(500)
def handle_500(err):
    if _wants_api():
        return jsonify({"error": "Internal server error. Please try again."}), 500
    return "500 — internal server error", 500


if __name__ == "__main__":
    # Debug mode (auto-reload) is OFF by default so the model only trains once.
    # During development you can enable it with:  FLASK_DEBUG=1 python app.py
    # (On Windows cmd:  set FLASK_DEBUG=1 && python app.py)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
