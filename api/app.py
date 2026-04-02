"""Simple Flask API for IPL winner prediction."""

from __future__ import annotations

import os

from flask import Flask, jsonify, request

from src.predict import predict_winner
from utils.config import MODEL_PATH

app = Flask(__name__)


@app.get("/health")
def health() -> tuple:
    """Health endpoint to verify service and model availability."""
    return (
        jsonify(
            {
                "status": "ok",
                "model_exists": os.path.exists(MODEL_PATH),
            }
        ),
        200,
    )


@app.post("/predict")
def predict() -> tuple:
    """Predict match winner for one or multiple records."""
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON payload."}), 400

    records = payload if isinstance(payload, list) else [payload]

    required = ["team1", "team2", "toss_winner", "toss_decision", "venue"]
    for index, record in enumerate(records):
        missing = [field for field in required if field not in record]
        if missing:
            return (
                jsonify({"error": f"Record {index} is missing fields: {missing}"}),
                400,
            )

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": f"Model file not found at {MODEL_PATH}"}), 404

    predictions = predict_winner(MODEL_PATH, records)
    return jsonify({"predictions": predictions}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
