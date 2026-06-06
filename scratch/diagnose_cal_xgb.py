import pandas as pd
import numpy as np
from src.data_fetcher import YFinanceProvider
from src.indicators import calculate_indicators
from src.features import prepare_data_for_training
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

fetcher = YFinanceProvider()

for symbol in ["SPY", "NIFTY"]:
    print(f"\n===== Calibrated XGBoost for {symbol} =====")
    df = fetcher.fetch_data(symbol)
    df = calculate_indicators(df)
    X, y = prepare_data_for_training(df)
    
    # Train-test split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Calibration split (80% fit, 20% calibrate)
    cal_split_idx = int(len(X_train) * 0.8)
    X_fit, X_cal = X_train.iloc[:cal_split_idx], X_train.iloc[cal_split_idx:]
    y_fit, y_cal = y_train.iloc[:cal_split_idx], y_train.iloc[cal_split_idx:]
    
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute scale_pos_weight
    num_neg = np.sum(y_fit == 0)
    num_pos = np.sum(y_fit == 1)
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    base_clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=12,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.15,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=15,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    base_clf.fit(X_fit_scaled, y_fit, eval_set=[(X_cal_scaled, y_cal)], verbose=False)
    
    # Sigmoid calibration on validation set
    cal_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv="prefit")
    cal_clf.fit(X_cal_scaled, y_cal)
    
    y_pred_proba = cal_clf.predict_proba(X_test_scaled)[:, 1]
    
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60]:
        y_pred = (y_pred_proba >= thresh).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        num_trades = np.sum(y_pred)
        print(f"Thresh={thresh:.2f}: Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, Trades: {num_trades}")
