"""
NBA Predictive Modeling - Model Training Pipeline
Author: Aleksandr Gvozdkov
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate NBA dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records with {df.shape[1]} features")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for NBA game prediction."""
    # Rolling averages for last 5 games
    for col in ['pts', 'reb', 'ast', 'fg_pct', 'ft_pct']:
        df[f'{col}_rolling_5'] = df.groupby('team_id')[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
    
    # Home/away performance differential
    df['home_advantage'] = df['is_home'].astype(int)
    
    # Win rate last 10 games
    df['win_rate_10'] = df.groupby('team_id')['win'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    
    # Head-to-head historical win rate
    df['h2h_win_rate'] = df.groupby(['team_id', 'opponent_id'])['win'].transform('mean')
    
    return df


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_label_encoder': False,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM classifier."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    return model


def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    
    print(f"\n=== {model_name} Results ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"AUC-ROC:  {auc_roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))
    
    return {'accuracy': accuracy, 'auc_roc': auc_roc}


def main():
    print("NBA Predictive Modeling — Training Pipeline")
    print("=" * 50)
    
    # Load data
    df = load_data('data/nba_games.csv')
    
    # Feature engineering
    df = engineer_features(df)
    
    # Define features and target
    feature_cols = [c for c in df.columns if c not in ['game_id', 'date', 'win', 'team_id', 'opponent_id']]
    X = df[feature_cols].fillna(0)
    y = df['win']
    
    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Train models
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    print("Training LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    
    # Evaluate
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    lgb_results = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
    
    # Save best model
    best_model = xgb_model if xgb_results['auc_roc'] >= lgb_results['auc_roc'] else lgb_model
    best_name = "XGBoost" if xgb_results['auc_roc'] >= lgb_results['auc_roc'] else "LightGBM"
    
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"\nBest model ({best_name}) saved to models/best_model.pkl")


if __name__ == '__main__':
    main()
