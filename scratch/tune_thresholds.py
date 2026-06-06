import pandas as pd
import numpy as np
from src.data_fetcher import YFinanceProvider
from src.indicators import calculate_indicators
from src.features import prepare_data_for_training
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

fetcher = YFinanceProvider()

for symbol in ["SPY", "NIFTY"]:
    print(f"\n===== Tuning Raw XGBoost for {symbol} =====")
    df = fetcher.fetch_data(symbol)
    df = calculate_indicators(df)
    X, y = prepare_data_for_training(df)
    
    # Train-test split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Validation split for early stopping and threshold tuning (last 20% of train)
    val_split_idx = int(len(X_train) * 0.8)
    X_fit, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_fit, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
    
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute scale_pos_weight
    num_neg = np.sum(y_fit == 0)
    num_pos = np.sum(y_fit == 1)
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    # Let's try different model complexities
    # We want to prevent memorization by setting high regularization and small trees.
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=3,                 # shallow tree to prevent overfitting
        learning_rate=0.015,         # slow learning rate
        min_child_weight=20,         # large leaf size requirement
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,               # high L1 regularization
        reg_lambda=10.0,             # high L2 regularization
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf.fit(X_fit_scaled, y_fit, eval_set=[(X_val_scaled, y_val)], verbose=False)
    
    # Predict probabilities on Val and Test
    val_proba = clf.predict_proba(X_val_scaled)[:, 1]
    test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    # Tune threshold on Val
    best_thresh = 0.5
    best_val_prec = 0.0
    
    # Search thresholds from 0.45 to 0.70
    thresholds = np.linspace(0.45, 0.70, 51)
    for t in thresholds:
        val_pred = (val_proba >= t).astype(int)
        rec = recall_score(y_val, val_pred, zero_division=0)
        prec = precision_score(y_val, val_pred, zero_division=0)
        
        # We want to maximize precision, but require at least 5% recall so we actually trade
        if rec >= 0.05:
            if prec > best_val_prec:
                best_val_prec = prec
                best_thresh = t
                
    print(f"Optimal Threshold found on Val: {best_thresh:.3f} (Val Precision: {best_val_prec:.4f})")
    
    # Evaluate on Test Set using optimal threshold
    test_pred_opt = (test_proba >= best_thresh).astype(int)
    test_acc = accuracy_score(y_test, test_pred_opt)
    test_prec = precision_score(y_test, test_pred_opt, zero_division=0)
    test_rec = recall_score(y_test, test_pred_opt, zero_division=0)
    num_trades = np.sum(test_pred_opt)
    
    print(f"Test Set with Optimal Threshold ({best_thresh:.3f}):")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  Trades:    {num_trades} (out of {len(y_test)})")
