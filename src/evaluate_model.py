"""Evaluation utilities for IPL models."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict:
    """Evaluate a classifier and return metrics payload."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    labels = sorted(y_test.unique().tolist())
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "labels": labels,
        "confusion_matrix": matrix,
    }


def print_evaluation_report(result: Dict) -> None:
    """Print metrics in a readable format."""
    print("\n" + "=" * 70)
    print(f"Model: {result['model_name']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    matrix_df = pd.DataFrame(
        result["confusion_matrix"], index=result["labels"], columns=result["labels"]
    )
    print(matrix_df)
