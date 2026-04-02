"""Feature engineering for IPL match winner prediction."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from utils.helpers import normalize_text


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


def _is_home_team(team: str, venue: str) -> int:
    team_key = normalize_text(team)
    venue_key = normalize_text(venue)
    keywords = TEAM_HOME_KEYWORDS.get(team_key, [])
    return int(any(keyword in venue_key for keyword in keywords))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create toss and venue-derived features from pre-match columns."""
    features = df.copy()

    features["team1_won_toss"] = (features["toss_winner"] == features["team1"]).astype(int)
    features["team2_won_toss"] = (features["toss_winner"] == features["team2"]).astype(int)

    features["team1_home_advantage"] = [
        _is_home_team(team, venue) for team, venue in zip(features["team1"], features["venue"])
    ]
    features["team2_home_advantage"] = [
        _is_home_team(team, venue) for team, venue in zip(features["team2"], features["venue"])
    ]

    # Toss advantage is represented by winner identity plus toss decision.
    features["toss_winner_is_team1"] = (features["toss_winner"] == features["team1"]).astype(int)
    features["toss_winner_is_team2"] = (features["toss_winner"] == features["team2"]).astype(int)

    return features
