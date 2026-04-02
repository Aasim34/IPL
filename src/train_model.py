"""Train and save IPL winner prediction models."""

from __future__ import annotations

import argparse
import os
from typing import Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_preprocessing import load_and_clean_data
from src.evaluate_model import evaluate_classifier, print_evaluation_report
from src.feature_engineering import add_engineered_features
from utils.config import MODEL_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH
from utils.helpers import ensure_directory


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing transformer for categorical and numeric features."""
    categorical_columns = ["team1", "team2", "toss_winner", "toss_decision", "venue"]
    numeric_columns = [
        "team1_won_toss",
        "team2_won_toss",
        "team1_home_advantage",
        "team2_home_advantage",
        "toss_winner_is_team1",
        "toss_winner_is_team2",
    ]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numeric", numeric_pipeline, numeric_columns),
        ]
    )


def build_models(preprocessor: ColumnTransformer, random_state: int) -> Dict[str, Pipeline]:
    """Create candidate model pipelines for comparison."""
    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "LogisticRegression": logistic_regression,
        "RandomForest": random_forest,
    }


def train(data_path: str, model_path: str, processed_path: str, test_size: float, random_state: int) -> None:
    """Full train/evaluate/save pipeline."""
    df = load_and_clean_data(data_path)
    df = add_engineered_features(df)

    ensure_directory(os.path.dirname(processed_path))
    df.to_csv(processed_path, index=False)

    feature_columns = [
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
        "team1_won_toss",
        "team2_won_toss",
        "team1_home_advantage",
        "team2_home_advantage",
        "toss_winner_is_team1",
        "toss_winner_is_team2",
    ]

    X = df[feature_columns]
    y = df["winner"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

    preprocessor = build_preprocessor()
    models = build_models(preprocessor, random_state)

    evaluations = []
    trained = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained[model_name] = model

        result = evaluate_classifier(model, X_test, y_test, model_name)
        evaluations.append(result)
        print_evaluation_report(result)

    comparison = pd.DataFrame(
        [{"model": result["model_name"], "accuracy": result["accuracy"]} for result in evaluations]
    ).sort_values(by="accuracy", ascending=False)

    print("\n" + "=" * 70)
    print("Model comparison:")
    print(comparison.to_string(index=False))

    best_name = comparison.iloc[0]["model"]
    best_accuracy = float(comparison.iloc[0]["accuracy"])
    best_model = trained[best_name]

    artifact = {
        "model_name": best_name,
        "accuracy": best_accuracy,
        "pipeline": best_model,
        "required_input_columns": feature_columns,
    }

    ensure_directory(os.path.dirname(model_path))
    joblib.dump(artifact, model_path)

    print("\n" + "=" * 70)
    print(f"Best model: {best_name} (accuracy={best_accuracy:.4f})")
    print(f"Model saved to: {model_path}")
    print(f"Cleaned data saved to: {processed_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IPL winner prediction model")
    parser.add_argument("--data-path", default=RAW_DATA_PATH, help="Input raw matches CSV path")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Output model artifact path")
    parser.add_argument(
        "--processed-path",
        default=PROCESSED_DATA_PATH,
        help="Output cleaned CSV path",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_path=args.data_path,
        model_path=args.model_path,
        processed_path=args.processed_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
