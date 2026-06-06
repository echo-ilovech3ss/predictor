import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from src.logger import logger

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class MarketMLModel:
    """Handles training, saving, loading, and predicting using a calibrated classifier."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol.upper().replace("^", "")
        self.model_path = os.path.join(MODEL_DIR, f"{self.symbol}_model.joblib")
        self.model = None
        self.scaler = None
        self.feature_names = []
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model using a time-series split.
        Applies strict scaling and calibration to prevent leakage.
        """
        logger.info(f"Starting model training for {self.symbol}...")
        
        # 1. Time-Series Train/Test Split (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 2. Calibration Split within Training Data (80% fit, 20% calibrate)
        cal_split_idx = int(len(X_train) * 0.8)
        X_fit, X_cal = X_train.iloc[:cal_split_idx], X_train.iloc[cal_split_idx:]
        y_fit, y_cal = y_train.iloc[:cal_split_idx], y_train.iloc[cal_split_idx:]
        
        self.feature_names = list(X.columns)
        
        # 3. Fit scaler ONLY on X_fit (training subset)
        self.scaler = StandardScaler()
        X_fit_scaled = self.scaler.fit_transform(X_fit)
        X_cal_scaled = self.scaler.transform(X_cal)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 4. Train base model on X_fit using XGBClassifier
        # Prioritize reducing False Positives (bad trades) and maximizing Precision.
        # Setting scale_pos_weight to 0.70 actively penalizes false positive errors during training.
        scale_pos_weight = 0.70
        logger.info(f"Prioritizing high precision. Using scale_pos_weight={scale_pos_weight:.2f}")
        
        base_clf = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.04,
            min_child_weight=10,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        base_clf.fit(X_fit_scaled, y_fit)
        
        # 5. Calibrate probabilities on X_cal
        # Using cv='prefit' allows calibrating a model that has already been fitted on a disjoint set.
        self.model = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv="prefit")
        self.model.fit(X_cal_scaled, y_cal)
        
        # 6. Evaluate on Test set
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"--- Model Evaluation for {self.symbol} ---")
        logger.info(f"Test Accuracy:  {acc:.4f}")
        logger.info(f"Test Precision: {prec:.4f}")
        logger.info(f"Test Recall:    {rec:.4f}")
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
            "confusion_matrix": cm
        }
        
    def save(self):
        """Save the trained model and scaler."""
        if self.model is None or self.scaler is None:
            logger.error("Cannot save model: not trained yet.")
            return
            
        data_to_save = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names
        }
        try:
            joblib.dump(data_to_save, self.model_path)
            logger.info(f"Model successfully saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            
    def load(self) -> bool:
        """Load model and scaler from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"No trained model found at {self.model_path}")
            return False
            
        try:
            saved_data = joblib.load(self.model_path)
            self.model = saved_data["model"]
            self.scaler = saved_data["scaler"]
            self.feature_names = saved_data["feature_names"]
            logger.info(f"Loaded trained model for {self.symbol} from {self.model_path}")
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
