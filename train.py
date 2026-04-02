"""
Training script for IPL match winner prediction.

This script:
1. Loads and validates IPL match data.
2. Applies feature engineering (toss advantage and home-ground signals).
3. Trains and evaluates two models (Logistic Regression and Random Forest).
4. Compares model performance.
5. Saves the best model artifact with joblib.

Usage:
    python train.py
    python train.py --data-path data/raw/matches.csv --model-path models/ipl_model.pkl
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REQUIRED_COLUMNS = ["team1", "team2", "toss_winner", "toss_decision", "venue", "winner"]

# Team to city/venue keywords for a lightweight home-ground proxy.
# Keywords are matched against normalized venue text.
TEAM_HOME_KEYWORDS: Dict[str, List[str]] = {
    "chennai super kings": ["chennai", "chepauk", "m. a. chidambaram"],
    "mumbai indians": ["mumbai", "wankhede", "brabourne", "dy patil"],
    "kolkata knight riders": ["kolkata", "eden gardens"],
    "royal challengers bangalore": ["bengaluru", "bangalore", "chinnaswamy"],
    "sunrisers hyderabad": ["hyderabad", "uppal", "rajiv gandhi"],
    "deccan chargers": ["hyderabad", "uppal", "rajiv gandhi"],
    "rajasthan royals": ["jaipur", "sawai mansingh"],
    "delhi capitals": ["delhi", "arun jaitley", "feroz shah kotla"],
    "delhi daredevils": ["delhi", "arun jaitley", "feroz shah kotla"],
    "punjab kings": ["mohali", "chandigarh", "dharamsala", "new chandigarh"],
    "kings xi punjab": ["mohali", "chandigarh", "dharamsala", "new chandigarh"],
    "lucknow super giants": ["lucknow", "ekana", "atal bihari vajpayee"],
    "gujarat titans": ["ahmedabad", "narendra modi", "motera"],
    "rising pune supergiant": ["pune", "maharashtra cricket association"],
    "rising pune supergiants": ["pune", "maharashtra cricket association"],
    "pune warriors": ["pune", "maharashtra cricket association"],
    "kochi tuskers kerala": ["kochi"],
    "gujarat lions": ["rajkot", "saurashtra cricket association"],
}


@dataclass
class TrainResult:
    name: str
    pipeline: Pipeline
    accuracy: float
    confusion_matrix: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IPL match winner prediction model")
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join("data", "raw", "matches.csv"),
        help="Path to input CSV dataset",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("models", "ipl_model.pkl"),
        help="Path to save best trained model artifact",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )


def filter_valid_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep matches where winner is one of the two participating teams."""
    winner = df["winner"].astype(str)
    valid_mask = winner.eq(df["team1"].astype(str)) | winner.eq(df["team2"].astype(str))
    return df.loc[valid_mask].copy()


def is_home_team(team: str, venue: str) -> int:
    team_key = normalize_text(team)
    venue_key = normalize_text(venue)
    keywords = TEAM_HOME_KEYWORDS.get(team_key, [])
    return int(any(keyword in venue_key for keyword in keywords))


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Toss advantage: team1 won toss and chose to bat/bowl first as captured by toss_decision.
    out["team1_won_toss"] = (out["toss_winner"] == out["team1"]).astype(int)
    out["team2_won_toss"] = (out["toss_winner"] == out["team2"]).astype(int)

    # Home-ground proxy features from venue text.
    out["team1_home_advantage"] = [
        is_home_team(team, venue) for team, venue in zip(out["team1"], out["venue"])
    ]
    out["team2_home_advantage"] = [
        is_home_team(team, venue) for team, venue in zip(out["team2"], out["venue"])
    ]

    # Explicit toss-decision interaction feature.
    out["toss_winner_is_team1"] = (out["toss_winner"] == out["team1"]).astype(int)
    out["toss_winner_is_team2"] = (out["toss_winner"] == out["team2"]).astype(int)

    return out


def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df = pd.read_csv(data_path)
    ensure_required_columns(df)

    # Basic cleaning for required columns.
    df = df.copy()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype("string").str.strip()

    # Remove rows where target is missing.
    df = df.dropna(subset=["winner"])
    df = df[df["winner"].str.len() > 0]

    # Keep only rows with valid winner labels and reduce noise from ties/no-results.
    df = filter_valid_outcomes(df)

    if df.empty:
        raise ValueError("No valid training rows left after cleaning/filtering.")

    df = add_feature_engineering(df)

    # Feature set focused on pre-match available information.
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

    return X, y


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )


def build_model_pipelines(preprocessor: ColumnTransformer, random_state: int) -> Dict[str, Pipeline]:
    pipelines = {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        multi_class="auto",
                        n_jobs=None,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return pipelines


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pipelines: Dict[str, Pipeline],
) -> List[TrainResult]:
    results: List[TrainResult] = []

    label_order = sorted(y_test.unique().tolist())

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds, labels=label_order)

        results.append(
            TrainResult(
                name=name,
                pipeline=pipeline,
                accuracy=acc,
                confusion_matrix=cm,
            )
        )

        print("\n" + "=" * 70)
        print(f"Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
        print(cm_df)

    return results


def save_best_model(results: List[TrainResult], model_path: str) -> TrainResult:
    best = max(results, key=lambda r: r.accuracy)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    artifact = {
        "model_name": best.name,
        "pipeline": best.pipeline,
        "accuracy": best.accuracy,
        "required_input_columns": [
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
        ],
        "note": "Pipeline contains preprocessing + classifier. Use pipeline.predict(input_df).",
    }

    joblib.dump(artifact, model_path)
    return best


def main() -> None:
    args = parse_args()

    print("Loading and preparing data...")
    X, y = load_and_prepare_data(args.data_path)

    categorical_cols = ["team1", "team2", "toss_winner", "toss_decision", "venue"]
    numeric_cols = [
        "team1_won_toss",
        "team2_won_toss",
        "team1_home_advantage",
        "team2_home_advantage",
        "toss_winner_is_team1",
        "toss_winner_is_team2",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    pipelines = build_model_pipelines(preprocessor, args.random_state)

    print("\nTraining and evaluating models...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test, pipelines)

    print("\n" + "=" * 70)
    print("Model comparison (sorted by accuracy):")
    comparison = pd.DataFrame(
        [{"model": r.name, "accuracy": r.accuracy} for r in results]
    ).sort_values(by="accuracy", ascending=False)
    print(comparison.to_string(index=False))

    best = save_best_model(results, args.model_path)
    print("\n" + "=" * 70)
    print(f"Best model: {best.name} (accuracy={best.accuracy:.4f})")
    print(f"Saved to: {args.model_path}")


if __name__ == "__main__":
    main()
