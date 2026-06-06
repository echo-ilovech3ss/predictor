import pandas as pd
import numpy as np
from src.data_fetcher import YFinanceProvider
from src.indicators import calculate_indicators
from src.features import prepare_data_for_training
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

fetcher = YFinanceProvider()

for symbol in ["SPY", "NIFTY"]:
    df = fetcher.fetch_data(symbol)
    df = calculate_indicators(df)
    X, y = prepare_data_for_training(df)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    val_split_idx = int(len(X_train) * 0.85)
    X_fit, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_fit, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
    
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    
    num_neg = np.sum(y_fit == 0)
    num_pos = np.sum(y_fit == 1)
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.02,
        min_child_weight=15,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=25,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf.fit(X_fit_scaled, y_fit, eval_set=[(scaler.transform(X_val), y_val)], verbose=False)
    print(f"{symbol} Best Iteration: {clf.best_iteration}, Total Trees: {clf.n_estimators}")
