# 🏀 NBA Predictive Modeling

An end-to-end machine learning pipeline for forecasting NBA game outcomes and
individual player performance metrics.  The system identifies statistical
inefficiencies in player prop markets and models accurate probabilities,
simulating real-world approaches used in quantitative sports analytics.

---

## 📁 Project Structure

```
nba-predictive-modeling/
├── src/
│   ├── data/
│   │   ├── scraper.py        # Box score & injury report scraping
│   │   ├── cleaner.py        # Data type coercion & column standardisation
│   │   └── aggregator.py     # Rolling features & back-to-back detection
│   ├── features/
│   │   └── engineering.py    # TS%, Pace, Off/Def/Net Rating
│   ├── models/
│   │   ├── game_predictor.py # Home-win probability (XGBoost + LGB + LR)
│   │   └── player_props.py   # Over/under classifier + market edge scanner
│   ├── evaluation/
│   │   └── metrics.py        # Log Loss, Brier Score, ROC-AUC
│   └── alerts/
│       └── telegram_bot.py   # Telegram Bot alert integration
├── scripts/
│   └── run_pipeline.py       # End-to-end orchestration
├── tests/                    # pytest test suite
├── data/
│   ├── raw/                  # Scraped CSVs (git-ignored)
│   └── processed/            # Cleaned CSVs (git-ignored)
├── models/                   # Serialised model artefacts (git-ignored)
├── requirements.txt
└── setup.py
```

---

## ⚙️ Core Features

### Statistical Inefficiency Detection
Uses player rolling averages (points, rebounds, assists), True Shooting %, and
game-context signals to estimate the probability of a player exceeding a prop
line.  The model then computes the **edge** vs. the market-implied probability
derived from American or decimal odds to identify value opportunities.

### Game Probability Modeling
Ensemble of **XGBoost**, **LightGBM**, and **Logistic Regression** models that
evaluate:
- Team rolling offensive output (10-game windows)
- Back-to-back scheduling flags
- Advanced metrics: Net Rating, Offensive/Defensive Rating, Pace
- Key injury report signals

### Automated Data Pipeline
- Scrapes daily NBA box scores and player game logs from Basketball-Reference.
- Scrapes the live injury report from ESPN.
- Cleans, validates, and persists data to `data/processed/`.

### Evaluation Metrics
Every model is evaluated with:
| Metric | Description |
|---|---|
| **Log Loss** | Cross-entropy; lower is better |
| **Brier Score** | Mean squared probability error; lower is better |
| **ROC-AUC** | Discrimination ability; higher is better |

---

## 🛠️ Tech Stack

| Layer | Libraries |
|---|---|
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Data Engineering | Python 3.10+, Pandas, NumPy |
| Web Scraping | requests, BeautifulSoup4, lxml |
| Alerts | python-telegram-bot |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (scrape → clean → train → evaluate)
python scripts/run_pipeline.py --season 2024

# 3. Optionally send a Telegram alert on completion
export TELEGRAM_BOT_TOKEN="<your-token>"
export TELEGRAM_CHAT_ID="<your-chat-id>"
python scripts/run_pipeline.py --season 2024 --alert
```

### Using the models directly

```python
from src.models.game_predictor import GamePredictor

predictor = GamePredictor()
predictor.fit(train_df)          # DataFrame with feature + target columns
probs = predictor.predict_proba(test_df)   # array of home-win probabilities
predictor.save()                 # persists to models/game_predictor.pkl
```

```python
from src.models.player_props import PlayerPropsModel

model = PlayerPropsModel(stat="points")
df = PlayerPropsModel.create_over_target(df, stat="points", line=24.5)
model.fit(df)

# Scan for value versus market odds
result = model.scan_for_value(df, lines=lines_series, market_odds=odds_series)
value_bets = result[result["value"] == True]
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🔮 Future Integration

- **Real-time adjustments** — live-game data stream integration for in-game
  probability recalculation.
- **Telegram Bot** — automated high-value deviation alerts already scaffolded
  in `src/alerts/telegram_bot.py`.
