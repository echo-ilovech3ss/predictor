import pandas as pd
import numpy as np
from src.data_fetcher import YFinanceProvider
from src.indicators import calculate_indicators
from src.features import prepare_data_for_training
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score

fetcher = YFinanceProvider()

for symbol in ["SPY", "NIFTY"]:
    print(f"\n===== OOF Threshold Tuning with scale_pos_weight=1.0 for {symbol} =====")
    df = fetcher.fetch_data(symbol)
    df = calculate_indicators(df)
    X, y = prepare_data_for_training(df)
    
    # Train-test split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 5-fold KFold (non-shuffled for time-series is better, or standard KFold is fine for OOF scaling)
    # Let's use 5-fold KFold without shuffling to preserve local temporal structures, or with shuffling for unbiased estimation.
    # Actually, standard KFold (shuffled) is best for out-of-fold calibration.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_probas = np.zeros(len(X_train))
    
    for train_cv_idx, val_cv_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_cv_idx], X_train.iloc[val_cv_idx]
        y_tr, y_val = y_train.iloc[train_cv_idx], y_train.iloc[val_cv_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        fold_clf = XGBClassifier(
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
        fold_clf.fit(X_tr_scaled, y_tr)
        
        val_proba = fold_clf.predict_proba(X_val_scaled)[:, 1]
        oof_probas[val_cv_idx] = val_proba
        
    # Find optimal threshold on OOF predictions
    best_t = 0.50
    best_oof_prec = 0.0
    
    # Let's optimize a metric that balances precision and recall:
    # e.g., maximize precision subject to recall >= 10%
    for t in np.linspace(0.50, 0.65, 31):
        preds = (oof_probas >= t).astype(int)
        rec = recall_score(y_train, preds, zero_division=0)
        prec = precision_score(y_train, preds, zero_division=0)
        
        if rec >= 0.10:  # require at least 10% recall
            if prec > best_oof_prec:
                best_oof_prec = prec
                best_t = t
                
    print(f"OOF Tuned Threshold: {best_t:.3f} (OOF Precision: {best_oof_prec:.4f})")
    
    # Train final model on full Train set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    final_clf = XGBClassifier(
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
    final_clf.fit(X_train_scaled, y_train)
    
    # Evaluate on Test set
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
