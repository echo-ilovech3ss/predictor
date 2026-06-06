import pandas as pd
import numpy as np
from src.data_fetcher import YFinanceProvider
from src.indicators import calculate_indicators
from src.features import prepare_data_for_training
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

fetcher = YFinanceProvider()

for symbol in ["SPY", "NIFTY"]:
    print(f"\n===== Diagnosing Raw XGBoost for {symbol} =====")
    df = fetcher.fetch_data(symbol)
    df = calculate_indicators(df)
    X, y = prepare_data_for_training(df)
    
    # Train-test split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Validation split for early stopping (e.g., last 15% of train)
    val_split_idx = int(len(X_train) * 0.85)
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
    
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,                # slightly shallower to prevent overfitting
        learning_rate=0.02,         # smaller learning rate
        min_child_weight=15,        # larger min_child_weight for regularization
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,              # higher L1 regularization
        reg_lambda=5.0,             # higher L2 regularization
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=25,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf.fit(X_fit_scaled, y_fit, eval_set=[(X_val_scaled, y_val)], verbose=False)
    
    # Evaluate raw classifier on test set
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Standard (threshold=0.5): Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Let's also evaluate with a higher trade threshold, say 0.55 or 0.60
    for thresh in [0.55, 0.60, 0.65]:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        acc_t = accuracy_score(y_test, y_pred_thresh)
        prec_t = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec_t = recall_score(y_test, y_pred_thresh, zero_division=0)
        num_trades = np.sum(y_pred_thresh)
        print(f"Threshold={thresh:.2f}: Acc: {acc_t:.4f}, Prec: {prec_t:.4f}, Rec: {rec_t:.4f}, Trades: {num_trades}")
