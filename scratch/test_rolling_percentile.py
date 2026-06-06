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
    print(f"\n===== Rolling Percentile Evaluation for {symbol} =====")
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
    clf.fit(X_train_scaled, y_train)
    
    # Predict probabilities on Train and Test
    train_proba = clf.predict_proba(X_train_scaled)[:, 1]
    test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    # Concatenate all probabilities to simulate a rolling stream
    all_proba = np.concatenate([train_proba, test_proba])
    all_y = np.concatenate([y_train, y_test])
    
    # Start evaluating from the beginning of the test set
    test_start_idx = len(train_proba)
    
    # Evaluate different rolling window sizes and percentile thresholds
    for window_size in [500]:
        for pct in [80, 85, 90, 95]:
            preds = []
            true_labels = []
            
            for idx in range(test_start_idx, len(all_proba)):
                # Get the past window of probabilities to compute percentile threshold
                past_probas = all_proba[idx - window_size : idx]
                current_proba = all_proba[idx]
                
                threshold = np.percentile(past_probas, pct)
                
                if current_proba >= threshold:
                    preds.append(1)
                else:
                    preds.append(0)
                    
                true_labels.append(all_y[idx])
                
            preds = np.array(preds)
            true_labels = np.array(true_labels)
            
            acc = accuracy_score(true_labels, preds)
            prec = precision_score(true_labels, preds, zero_division=0)
            rec = recall_score(true_labels, preds, zero_division=0)
            trades = np.sum(preds)
            
            print(f"Window={window_size} | Pct={pct}% | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | Trades={trades}")
