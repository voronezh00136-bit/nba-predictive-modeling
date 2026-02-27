# 🏀 NBA Predictive Analytics & Probability Model

An end-to-end machine learning pipeline for forecasting NBA game outcomes and
individual player performance metrics.  It identifies statistical inefficiencies
in player-prop markets and models accurate win probabilities, simulating
real-world quantitative sports-analytics environments.

## Repository Layout

```
nba-predictive-modeling/
├── data/
│   └── pipeline.py        # Scraping, cleaning & aggregation helpers
├── models/
│   ├── player_props.py    # Points / rebounds / assists regression models
│   └── game_probability.py # Calibrated win-probability classifier
├── evaluation/
│   └── metrics.py         # Log Loss, Brier Score, ROC-AUC wrappers
├── tests/                 # pytest test suite
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Core Features

### Data Pipeline (`data/pipeline.py`)
* `clean_box_scores(raw)` — validates, casts, and enriches raw box-score data
  with True Shooting % and a Pace proxy.
* `aggregate_player_stats(box_scores, window)` — rolling per-player averages
  and standard deviations over the last *n* games.
* `build_game_features(box_scores, schedule)` — team-level feature matrix
  with back-to-back scheduling flags.

### Player Props Model (`models/player_props.py`)
* One gradient-boosting regressor (XGBoost → LightGBM → sklearn fallback)
  per prop target: **points**, **rebounds**, **assists**.
* `find_inefficiencies(X, market_lines, threshold)` — returns rows where the
  model's projection deviates from posted market lines by at least *threshold*
  units, labelled `over` / `under`.

### Game Probability Model (`models/game_probability.py`)
* Gradient-boosting binary classifier wrapped in isotonic-regression
  calibration for well-calibrated win probabilities.
* Learns from team advanced metrics (True Shooting %, Pace, Defensive Rating),
  back-to-back flags, and home/away context.

### Evaluation (`evaluation/metrics.py`)
* `compute_log_loss`, `compute_brier_score`, `compute_roc_auc` — individual
  wrappers around scikit-learn scorers.
* `evaluate_predictions(y_true, y_prob)` — single call returning all three
  metrics as a dictionary.

## Tech Stack
* **Machine Learning:** Scikit-learn, XGBoost, LightGBM
* **Data Engineering:** Python, Pandas, NumPy
* **Evaluation:** Log Loss, Brier Score, ROC-AUC

## Future Integration
* Real-time probability adjustments from live-game data streams.
* Telegram Bot API alerts for high-value statistical deviations.
