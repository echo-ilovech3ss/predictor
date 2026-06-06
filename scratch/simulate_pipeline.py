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
    print(f"\n===== Simulating Training Pipeline for {symbol} =====")
    df = fetcher.fetch_data(symbol)
    df = calculate_indicators(df)
    X, y = prepare_data_for_training(df)
    
    # Train-test split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Validation split for threshold tuning (last 20% of train)
    val_split_idx = int(len(X_train) * 0.8)
    X_fit, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_fit, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
    
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    num_neg = np.sum(y_fit == 0)
    num_pos = np.sum(y_fit == 1)
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    # Using robust regularization
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=15,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf.fit(X_fit_scaled, y_fit)
    
    # Predict probabilities on Val
    val_proba = clf.predict_proba(X_val_scaled)[:, 1]
    
    # Find optimal threshold on Val that maximizes precision with recall >= 5%
    best_t = 0.50
    best_val_prec = 0.0
    for t in np.linspace(0.50, 0.70, 41):
        preds = (val_proba >= t).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        if rec >= 0.05:
            if prec > best_val_prec:
                best_val_prec = prec
                best_t = t
                
    print(f"Tuned Threshold on Val: {best_t:.3f} (Val Precision: {best_val_prec:.4f})")
    
    # Evaluate on Test set using tuned threshold
    test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_proba >= best_t).astype(int)
    
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_trades = np.sum(test_preds)
    
    print(f"Test Set Evaluation with Threshold {best_t:.3f}:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  Trades:    {test_trades} (out of {len(y_test)})")
