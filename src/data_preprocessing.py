"""Data loading and preprocessing for IPL match winner prediction."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from utils.config import REQUIRED_COLUMNS


def ensure_required_columns(df: pd.DataFrame) -> None:
    """Raise an error if any required dataset columns are missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )


def filter_valid_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where winner is one of the two playing teams."""
    valid_mask = (df["winner"] == df["team1"]) | (df["winner"] == df["team2"])
    return df.loc[valid_mask].copy()


def clean_matches_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing steps to make data model-ready."""
    ensure_required_columns(df)

    cleaned = df.copy()
    for column in REQUIRED_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    cleaned = cleaned.dropna(subset=["winner"])
    cleaned = cleaned[cleaned["winner"].str.len() > 0]
    cleaned = filter_valid_outcomes(cleaned)

    if cleaned.empty:
        raise ValueError("No valid rows left after preprocessing.")

    return cleaned


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Read CSV from disk and apply cleaning pipeline."""
    dataframe = pd.read_csv(path)
    return clean_matches_data(dataframe)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split cleaned dataframe into feature matrix and target vector."""
    feature_columns = ["team1", "team2", "toss_winner", "toss_decision", "venue"]
    X = df[feature_columns].copy()
    y = df["winner"].astype(str).copy()
    return X, y
