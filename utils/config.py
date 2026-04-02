"""Central configuration values for paths and dataset schema."""

from __future__ import annotations

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "matches.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_data.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ipl_model.pkl")

REQUIRED_COLUMNS = ["team1", "team2", "toss_winner", "toss_decision", "venue", "winner"]
