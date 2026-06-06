import pandas as pd
import numpy as np

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a given OHLCV DataFrame.
    Returns a copy of the DataFrame with the new indicators appended.
    """
    # Create a copy to prevent SettingWithCopyWarnings
    data = df.copy()
    
    if len(data) < 200:
        # We need at least 200 data points to calculate EMA 200 and other rolling windows
        return data
        
    # 1. EMAs (20, 50, 200)
    data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['ema_200'] = data['close'].ewm(span=200, adjust=False).mean()
    
    # 2. RSI (14) using Wilder's smoothing technique
    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    
    # Avoid division by zero
    rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
    data['rsi'] = 100 - (100 / (1 + rs))
    # Fill any NaN in RSI with 50 (neutral)
    data['rsi'] = data['rsi'].fillna(50)
    
    # 3. MACD (12, 26, 9)
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # 4. Bollinger Bands (20, 2)
    bb_middle = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = bb_middle + 2 * bb_std
    data['bb_lower'] = bb_middle - 2 * bb_std
    data['bb_middle'] = bb_middle
    
    # 5. ATR (14) - Average True Range
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = (data['high'] - prev_close).abs()
    tr3 = (data['low'] - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # 6. Volume Change and Volume MA Ratio
    data['volume_change'] = data['volume'].pct_change()
    volume_ma = data['volume'].rolling(window=20).mean()
    data['volume_ma_ratio'] = np.where(volume_ma == 0, 1.0, data['volume'] / volume_ma)
    
    # 7. Returns
    data['returns'] = data['close'].pct_change()
    
    # 8. Rolling Volatility (20-period)
    data['volatility_20'] = data['returns'].rolling(window=20).std()
    
    # 9. ROC (Rate of Change) - 12-period
    data['roc_12'] = data['close'].pct_change(periods=12).fillna(0)
    
    # 10. Stochastic Oscillator (14, 3)
    lowest_low = data['low'].rolling(window=14).min()
    highest_high = data['high'].rolling(window=14).max()
    high_low_range = highest_high - lowest_low
    high_low_range = np.where(high_low_range == 0, 1.0, high_low_range)
    
    data['stoch_k'] = ((data['close'] - lowest_low) / high_low_range) * 100.0
    data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
    data['stoch_k'] = data['stoch_k'].fillna(50.0)
    data['stoch_d'] = data['stoch_d'].fillna(50.0)
    
    return data
