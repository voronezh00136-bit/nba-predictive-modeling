# 🏀 NBA Predictive Modeling

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

> ML pipeline for NBA player performance forecasting and game probability modeling with **78%+ accuracy**.

---

## 📊 Overview

This project builds an end-to-end machine learning pipeline to:
- Forecast **NBA player performance** metrics (points, assists, rebounds)
- Predict **game outcomes** and win probability
- Provide feature importance analysis for sports analytics insights

---

## 🎯 Results

| Model | Accuracy | AUC-ROC |
|-------|----------|----------|
| XGBoost | **78.3%** | 0.84 |
| LightGBM | **77.9%** | 0.83 |
| Baseline (Random) | 50.0% | 0.50 |

---

## ⚡ Tech Stack

- **Python 3.10+**
- **XGBoost** & **LightGBM** — gradient boosting models
- **Scikit-learn** — preprocessing, cross-validation, metrics
- **Pandas** & **NumPy** — data manipulation
- **Matplotlib** & **Seaborn** — visualization

---

## 📁 Project Structure

```
nba-predictive-modeling/
├── data/                  # Raw and processed datasets
├── notebooks/             # EDA and modeling notebooks
├── src/                   # Source code
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and evaluation
│   └── utils/             # Helper functions
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/voronezh00136-bit/nba-predictive-modeling.git
cd nba-predictive-modeling
pip install -r requirements.txt
```

---

## 👤 Author

**Aleksandr Gvozdkov** — ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aleksandr-gvozdkov/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/voronezh00136-bit)