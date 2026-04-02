# IPL ML Project

End-to-end machine learning project to predict IPL match winners from pre-match information.

## Project Structure

- data/raw/matches.csv: Raw IPL match data
- data/processed/cleaned_data.csv: Cleaned and feature-enriched data
- notebooks/eda.ipynb: Exploratory data analysis notebook
- src/: Core ML modules (preprocessing, feature engineering, training, evaluation, prediction)
- models/ipl_model.pkl: Saved best model artifact
- api/app.py: Flask inference API

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train Model

```bash
python -m src.train_model
```

Optional arguments:

```bash
python -m src.train_model --data-path data/raw/matches.csv --model-path models/ipl_model.pkl --processed-path data/processed/cleaned_data.csv
```

## Run API

```bash
python api/app.py
```

Health check:

```bash
GET http://localhost:8000/health
```

Prediction endpoint:

```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "team1": "Mumbai Indians",
  "team2": "Chennai Super Kings",
  "toss_winner": "Mumbai Indians",
  "toss_decision": "bat",
  "venue": "Wankhede Stadium"
}
```
