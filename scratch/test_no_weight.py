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
    print(f"\n===== XGBoost with scale_pos_weight=1.0 for {symbol} =====")
    df = fetcher.fetch_data(symbol)
    df = calculate_indicators(df)
    X, y = prepare_data_for_training(df)
    
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
        scale_pos_weight=1.0,  # No class weighting
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf.fit(X_train_scaled, y_train)
    
    test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    print("Threshold | Accuracy | Precision | Recall | Trades")
    print("------------------------------------------------")
    for t in np.linspace(0.48, 0.62, 15):
        test_pred = (test_proba >= t).astype(int)
        acc = accuracy_score(y_test, test_pred)
        prec = precision_score(y_test, test_pred, zero_division=0)
        rec = recall_score(y_test, test_pred, zero_division=0)
        trades = np.sum(test_pred)
        print(f"  {t:.2f}    |  {acc:.4f}  |  {prec:.4f}   | {rec:.4f} | {trades}")
