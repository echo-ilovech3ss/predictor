import pandas as pd
import numpy as np
from src.data_fetcher import YFinanceProvider
from src.indicators import calculate_indicators
from src.features import prepare_data_for_training
from src.ml_model import MarketMLModel
from src.logger import logger
import logging

# Set logger to print
logger.setLevel(logging.INFO)

# Fetch data for SPY
fetcher = YFinanceProvider()
df = fetcher.fetch_data("SPY")
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

# Fit scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_fit_scaled = scaler.fit_transform(X_fit)
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)

# Train model
from xgboost import XGBClassifier
num_neg = np.sum(y_fit == 0)
num_pos = np.sum(y_fit == 1)
scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

clf = XGBClassifier(
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
clf.fit(X_fit_scaled, y_fit, eval_set=[(X_cal_scaled, y_cal)], verbose=False)

# Check train/val/test performance
from sklearn.metrics import accuracy_score, precision_score, recall_score

for name, X_s, y_s in [("Fit", X_fit_scaled, y_fit), ("Cal", X_cal_scaled, y_cal), ("Test", X_test_scaled, y_test)]:
    preds = clf.predict(X_s)
    acc = accuracy_score(y_s, preds)
    prec = precision_score(y_s, preds, zero_division=0)
    rec = recall_score(y_s, preds, zero_division=0)
    print(f"{name} Set - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

# Also check with calibrated classifier
from sklearn.calibration import CalibratedClassifierCV
cal_clf = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
cal_clf.fit(X_cal_scaled, y_cal)

for name, X_s, y_s in [("Fit (Calibrated)", X_fit_scaled, y_fit), ("Cal (Calibrated)", X_cal_scaled, y_cal), ("Test (Calibrated)", X_test_scaled, y_test)]:
    preds = cal_clf.predict(X_s)
    acc = accuracy_score(y_s, preds)
    prec = precision_score(y_s, preds, zero_division=0)
    rec = recall_score(y_s, preds, zero_division=0)
    print(f"{name} Set - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")
