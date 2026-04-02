"""Prediction utilities for the trained IPL model artifact."""

from __future__ import annotations

from typing import Dict, List

import joblib
import pandas as pd

from src.feature_engineering import add_engineered_features


def load_artifact(model_path: str) -> Dict:
    """Load saved model artifact from disk."""
    return joblib.load(model_path)


def predict_winner(model_path: str, records: List[Dict]) -> List[str]:
    """Predict winners for a list of match records."""
    artifact = load_artifact(model_path)
    pipeline = artifact["pipeline"]

    input_df = pd.DataFrame(records)
    input_df = add_engineered_features(input_df)

    columns = artifact["required_input_columns"]
    predictions = pipeline.predict(input_df[columns])
    return predictions.tolist()
