import pandas as pd
import numpy as np
from src.logger import logger

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ML features from OHLCV and technical indicators.
    All features at index t are computed using data available at or before index t.
    """
    data = df.copy()
    
    # Check if necessary indicator columns are present
    required_indicators = ['ema_20', 'ema_50', 'ema_200', 'rsi', 'macd', 'bb_upper', 'atr']
    missing = [col for col in required_indicators if col not in data.columns]
    if missing:
        raise ValueError(f"Indicators must be calculated before feature extraction. Missing: {missing}")
        
    features = pd.DataFrame(index=data.index)
    
    # 1. Price distance to EMAs
    features['dist_ema_20'] = (data['close'] - data['ema_20']) / data['ema_20']
    features['dist_ema_50'] = (data['close'] - data['ema_50']) / data['ema_50']
    features['dist_ema_200'] = (data['close'] - data['ema_200']) / data['ema_200']
    
    # 2. EMA crossover relative distances
    features['ema_20_50'] = (data['ema_20'] - data['ema_50']) / data['ema_50']
    features['ema_50_200'] = (data['ema_50'] - data['ema_200']) / data['ema_200']
    
    # 3. RSI
    features['rsi'] = data['rsi']
    features['rsi_overbought'] = (data['rsi'] > 70).astype(float)
    features['rsi_oversold'] = (data['rsi'] < 30).astype(float)
    
    # 4. MACD relative to close price
    features['macd_norm'] = data['macd'] / data['close']
    features['macd_signal_norm'] = data['macd_signal'] / data['close']
    features['macd_hist_norm'] = data['macd_hist'] / data['close']
    
    # 5. Bollinger Bands distance
    features['dist_bb_upper'] = (data['bb_upper'] - data['close']) / data['close']
    features['dist_bb_lower'] = (data['close'] - data['bb_lower']) / data['close']
    features['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # 6. Volatility and ATR
    features['atr_norm'] = data['atr'] / data['close']
    features['volatility_20'] = data['volatility_20'].fillna(0)
    
    # 7. Volume dynamics
    features['volume_change'] = data['volume_change'].fillna(0)
    features['volume_ma_ratio'] = data['volume_ma_ratio']
    
    # 8. Lagged returns (information up to t)
    # returns represents pct_change from t-1 to t.
    features['returns_t'] = data['returns'].fillna(0)
    features['returns_t_1'] = data['returns'].shift(1).fillna(0)
    features['returns_t_2'] = data['returns'].shift(2).fillna(0)
    features['returns_t_3'] = data['returns'].shift(3).fillna(0)
    
    # 9. Candle characteristics (relative to ATR to scale across price levels)
    # Avoid division by zero in case ATR is 0
    safe_atr = np.where(data['atr'] == 0, 1.0, data['atr'])
    features['candle_body'] = (data['close'] - data['open']) / safe_atr
    features['candle_upper_shadow'] = (data['high'] - data[['open', 'close']].max(axis=1)) / safe_atr
    features['candle_lower_shadow'] = (data[['open', 'close']].min(axis=1) - data['low']) / safe_atr
    
    # 10. Intraday Session Seasonality
    features['hour_of_day'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    
    # 11. Cross-Market Features (only present if NIFTY has loaded aligned SPY features)
    if 'spy_returns' in data.columns:
        features['spy_returns_lag'] = data['spy_returns']
        features['spy_close_dist'] = (data['close'] - data['spy_close']) / data['spy_close']
        
    # 12. Momentum & Oscillator Features
    features['roc_12'] = data['roc_12']
    features['stoch_k'] = data['stoch_k']
    features['stoch_d'] = data['stoch_d']
    
    # Clean any inf or -inf values that could occur from divisions or percentage changes
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return features

def prepare_data_for_training(df: pd.DataFrame):
    """
    Generate features and labels for training.
    Label at index t is: 1 if Close(t+4) > Close(t) * 1.0005 (rises >= 0.05% over next 4 hours), else 0.
    Drops rows with missing features or target.
    """
    # Extract features
    features_df = extract_features(df)
    
    # Target label: Close(t+4) > Close(t) * 1.0005 (4-hour forward lookahead, 0.05% threshold)
    target = (df['close'].shift(-4) > df['close'] * 1.0005).astype(int)
    
    # Merge features and target
    dataset = features_df.copy()
    dataset['target'] = target
    
    # Drop rows with NaN (first 200 rows due to EMA200, and last 4 rows due to target shift)
    cleaned_dataset = dataset.dropna()
    
    X = cleaned_dataset.drop(columns=['target'])
    y = cleaned_dataset['target']
    
    logger.info(f"Prepared training dataset: X shape {X.shape}, y shape {y.shape}")
    return X, y
