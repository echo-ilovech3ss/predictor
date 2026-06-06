import pandas as pd
import numpy as np
from src.data_fetcher import YFinanceProvider
from src.indicators import calculate_indicators
from src.features import prepare_data_for_training
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score

fetcher = YFinanceProvider()

for symbol in ["SPY", "NIFTY"]:
    print(f"\n===== Out-Of-Fold Tuning for {symbol} =====")
    df = fetcher.fetch_data(symbol)
    df = calculate_indicators(df)
    X, y = prepare_data_for_training(df)
    
    # Train-test split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 5-fold TimeSeriesSplit on Train set to get Out-of-Fold probabilities
    tscv = TimeSeriesSplit(n_splits=5)
    oof_probas = np.zeros(len(X_train))
    oof_mask = np.zeros(len(X_train), dtype=bool)
    
    for train_cv_idx, val_cv_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_cv_idx], X_train.iloc[val_cv_idx]
        y_tr, y_val = y_train.iloc[train_cv_idx], y_train.iloc[val_cv_idx]
        
        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        # Train fold model
        num_neg = np.sum(y_tr == 0)
        num_pos = np.sum(y_tr == 1)
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        fold_clf = XGBClassifier(
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
        fold_clf.fit(X_tr_scaled, y_tr)
        
        # Predict on validation fold
        val_proba = fold_clf.predict_proba(X_val_scaled)[:, 1]
        oof_probas[val_cv_idx] = val_proba
        oof_mask[val_cv_idx] = True
        
    # Tune threshold on OOF predictions
    oof_y = y_train[oof_mask]
    oof_p = oof_probas[oof_mask]
    
    best_t = 0.50
    best_oof_prec = 0.0
    
    for t in np.linspace(0.50, 0.70, 41):
        preds = (oof_p >= t).astype(int)
        rec = recall_score(oof_y, preds, zero_division=0)
        prec = precision_score(oof_y, preds, zero_division=0)
        trades = np.sum(preds)
        
        # We want to maximize precision, requiring at least 3% recall (trades are selective but frequent enough)
        if rec >= 0.03:
            if prec > best_oof_prec:
                best_oof_prec = prec
                best_t = t
                
    print(f"OOF Tuned Threshold: {best_t:.3f} (OOF Precision: {best_oof_prec:.4f})")
    
    # Now train final model on full Train set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    final_clf = XGBClassifier(
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
    final_clf.fit(X_train_scaled, y_train)
    
    # Evaluate on Test set using OOF tuned threshold
    test_proba = final_clf.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_proba >= best_t).astype(int)
    
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_trades = np.sum(test_preds)
    
    print(f"Test Set Evaluation with OOF Threshold {best_t:.3f}:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  Trades:    {test_trades} (out of {len(y_test)})")
