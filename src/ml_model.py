import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from src.logger import logger

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class MarketMLModel:
    """Handles training, saving, loading, and predicting using a regularized XGBoost model."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol.upper().replace("^", "")
        self.model_path = os.path.join(MODEL_DIR, f"{self.symbol}_model.joblib")
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.optimal_threshold = 0.60
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model using a time-series split.
        Applies out-of-fold threshold tuning to prevent lookahead and optimize precision.
        """
        logger.info(f"Starting model training for {self.symbol}...")
        
        # 1. Time-Series Train/Test Split (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.feature_names = list(X.columns)
        
        # Fit scaler on X_train
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 2. Run KFold cross-validation on Training Set to find optimal decision threshold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_probas = np.zeros(len(X_train))
        
        logger.info("Running 5-fold cross-validation on training set to tune decision threshold...")
        for train_cv_idx, val_cv_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_cv_idx], X_train.iloc[val_cv_idx]
            y_tr, y_val = y_train.iloc[train_cv_idx], y_train.iloc[val_cv_idx]
            
            # Local scaling
            fold_scaler = StandardScaler()
            X_tr_scaled = fold_scaler.fit_transform(X_tr)
            X_val_scaled = fold_scaler.transform(X_val)
            
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
            oof_probas[val_cv_idx] = fold_clf.predict_proba(X_val_scaled)[:, 1]
            
        # Tune threshold on OOF predictions (maximize precision subject to recall >= 10%)
        best_t = 0.50
        best_oof_prec = 0.0
        
        for t in np.linspace(0.50, 0.65, 31):
            preds = (oof_probas >= t).astype(int)
            rec = recall_score(y_train, preds, zero_division=0)
            prec = precision_score(y_train, preds, zero_division=0)
            
            if rec >= 0.10:  # Require at least 10% trade frequency in validation folds
                if prec > best_oof_prec:
                    best_oof_prec = prec
                    best_t = t
                    
        self.optimal_threshold = float(best_t)
        logger.info(f"OOF Threshold Tuning completed. Optimal Threshold: {self.optimal_threshold:.3f} (OOF Precision: {best_oof_prec:.4f})")
        
        # 3. Train final model on full training set
        logger.info("Training final model on full training set...")
        self.model = XGBClassifier(
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
        self.model.fit(X_train_scaled, y_train)
        
        # 4. Evaluate on Test set using optimized threshold
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"--- Model Evaluation for {self.symbol} ---")
        logger.info(f"Test Accuracy (optimal threshold {self.optimal_threshold:.3f}):  {acc:.4f}")
        logger.info(f"Test Precision (optimal threshold {self.optimal_threshold:.3f}): {prec:.4f}")
        logger.info(f"Test Recall (optimal threshold {self.optimal_threshold:.3f}):    {rec:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(
            "CRITICAL NOTE: Accuracy is just one piece of the puzzle. "
            "Predictive accuracy in trading does not guarantee profitability. "
            "A model with 52% accuracy but a strong risk-reward ratio can be highly profitable, "
            "while a 60% accurate model with poor risk management can lose capital rapidly."
        )
        
        # Save model dictionary
        self.save()
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "confusion_matrix": cm,
            "optimal_threshold": self.optimal_threshold
        }
        
    def save(self):
        """Save the trained model, scaler, and dynamic threshold."""
        if self.model is None or self.scaler is None:
            logger.error("Cannot save model: not trained yet.")
            return
            
        data_to_save = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "optimal_threshold": self.optimal_threshold
        }
        try:
            joblib.dump(data_to_save, self.model_path)
            logger.info(f"Model successfully saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            
    def load(self) -> bool:
        """Load model, scaler, and dynamic threshold from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"No trained model found at {self.model_path}")
            return False
            
        try:
            saved_data = joblib.load(self.model_path)
            self.model = saved_data["model"]
            self.scaler = saved_data["scaler"]
            self.feature_names = saved_data["feature_names"]
            self.optimal_threshold = saved_data.get("optimal_threshold", 0.60)
            logger.info(f"Loaded trained model for {self.symbol} from {self.model_path}. Optimal Threshold: {self.optimal_threshold:.3f}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            return False
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of upward movement.
        Returns a tuple of (prob_down, prob_up).
        """
        if self.model is None or self.scaler is None:
            # Try to load
            if not self.load():
                raise ValueError("Model is not loaded or trained. Train the model first.")
                
        # Ensure we filter X to contain only the trained feature names in the same order
        X_filtered = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_filtered)
        
        # Predict probability
        # predict_proba returns [prob_class_0, prob_class_1]
        probas = self.model.predict_proba(X_scaled)
        return probas[0]  # returns array of [prob_down, prob_up] for the input row
