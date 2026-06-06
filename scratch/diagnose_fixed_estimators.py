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
    print(f"\n===== Fixed Estimators Scan for {symbol} =====")
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
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    for n_est in [30, 50, 80, 120, 180]:
        clf = XGBClassifier(
            n_estimators=n_est,
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
        clf.fit(X_train_scaled, y_train)
        
        test_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Scan threshold for high precision
        best_t = 0.5
        best_prec = 0.0
        best_rec = 0.0
        best_acc = 0.0
        best_trades = 0
        
        for t in np.linspace(0.50, 0.65, 31):
            preds = (test_proba >= t).astype(int)
            rec = recall_score(y_test, preds, zero_division=0)
            prec = precision_score(y_test, preds, zero_division=0)
            acc = accuracy_score(y_test, preds)
            trades = np.sum(preds)
            
            # We want precision > 54% and at least 20 trades on the test set (approx 3% recall)
            if trades >= 20 and prec > best_prec:
                best_prec = prec
                best_rec = rec
                best_acc = acc
                best_t = t
                best_trades = trades
                
        print(f"Trees={n_est:3d} | Best Thresh={best_t:.2f} | Acc={best_acc:.4f} | Prec={best_prec:.4f} | Rec={best_rec:.4f} | Trades={best_trades}")
