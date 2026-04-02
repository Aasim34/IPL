"""Utility helper functions used across the IPL project."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


def ensure_directory(path: str) -> None:
    """Create directory path if it does not exist."""
    os.makedirs(path, exist_ok=True)


def normalize_text(value: Any) -> str:
    """Normalize text for robust comparisons in feature engineering."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()
