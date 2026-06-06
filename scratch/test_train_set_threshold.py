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
    print(f"\n===== Train Set Threshold Tuning with scale_pos_weight=1.0 for {symbol} =====")
    df = fetcher.fetch_data(symbol)
    df = calculate_indicators(df)
    X, y = prepare_data_for_training(df)
    
    # Train-test split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=15,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=5.0,
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf.fit(X_train_scaled, y_train)
    
    # Predict probabilities on Train Set
    train_proba = clf.predict_proba(X_train_scaled)[:, 1]
    
    # Find optimal threshold on Train Set
    best_t = 0.50
    best_train_prec = 0.0
    
    # Scan thresholds from 0.50 to 0.65
    for t in np.linspace(0.50, 0.65, 31):
        preds = (train_proba >= t).astype(int)
        rec = recall_score(y_train, preds, zero_division=0)
        prec = precision_score(y_train, preds, zero_division=0)
        trades = np.sum(preds)
        
        # Require a minimum recall of 5% on the training set
        if rec >= 0.05:
            if prec > best_train_prec:
                best_train_prec = prec
                best_t = t
                
    print(f"Optimal Threshold on Train Set: {best_t:.3f} (Train Precision: {best_train_prec:.4f})")
    
    # Evaluate on Test Set
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
