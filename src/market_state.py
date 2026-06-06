import pandas as pd
import numpy as np
from src.logger import logger

def classify_market_state_row(row) -> str:
    """
    Classify the market state for a single data row (or Series).
    Returns one of: 'bullish', 'bearish', 'sideways', 'volatile', 'uncertain'
    """
    try:
        close = row['close']
        ema_20 = row['ema_20']
        ema_50 = row['ema_50']
        ema_200 = row['ema_200']
        rsi = row['rsi']
        macd = row['macd']
        macd_signal = row['macd_signal']
        atr = row['atr']
        bb_upper = row['bb_upper']
        bb_lower = row['bb_lower']
        bb_middle = row['bb_middle']
        volatility = row['volatility_20']
    except KeyError as e:
        # If columns are missing, return uncertain
        return "uncertain"

    # 1. Volatile Check
    # High volatility defined as ATR relative to price or high rolling volatility
    # Since we don't have historical series in a single row, we can check basic thresholds.
    # Typically, if the standard deviation of returns is high or price is far outside BB.
    is_volatile = False
    bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
    if volatility > 0.015 or bb_width > 0.05:  # e.g., 1.5% hourly volatility or 5% BB width
        is_volatile = True

    # 2. Bullish Check
    # Price is above key EMAs, EMAs are stacked, MACD is positive, RSI > 50
    is_bullish = (
        close > ema_50 and 
        ema_50 > ema_200 and 
        macd > macd_signal and 
        rsi > 50
    )

    # 3. Bearish Check
    # Price is below key EMAs, EMAs are stacked down, MACD is negative, RSI < 50
    is_bearish = (
        close < ema_50 and 
        ema_50 < ema_200 and 
        macd < macd_signal and 
        rsi < 50
    )

    # 4. Sideways Check
    # Narrow BB, RSI neutral, price crossing moving averages
    is_sideways = (
        bb_width < 0.018 and  # Compressed bands
        40 <= rsi <= 60 and
        abs(close - bb_middle) / bb_middle < 0.01
    )

    # Resolve classification priority
    if is_volatile:
        # If volatile but strongly trending, keep trend class or flag volatile
        if is_bullish:
            return "bullish"
        elif is_bearish:
            return "bearish"
        else:
            return "volatile"
    elif is_bullish:
        return "bullish"
    elif is_bearish:
        return "bearish"
    elif is_sideways:
        return "sideways"
    else:
        return "uncertain"

def classify_market_states(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized or iterative classification of market states across a whole DataFrame.
    """
    if df.empty:
        return pd.Series(dtype=str)
        
    states = []
    # Using itertuples is much faster than apply for large DataFrames
    for row in df.itertuples():
        # Map named tuple back to dictionary-like fields for classify_market_state_row
        row_dict = {
            'close': row.close,
            'ema_20': row.ema_20,
            'ema_50': row.ema_50,
            'ema_200': row.ema_200,
            'rsi': row.rsi,
            'macd': row.macd,
            'macd_signal': row.macd_signal,
            'atr': row.atr,
            'bb_upper': row.bb_upper,
            'bb_lower': row.bb_lower,
            'bb_middle': row.bb_middle,
            'volatility_20': row.volatility_20
        }
        states.append(classify_market_state_row(row_dict))
        
    return pd.Series(states, index=df.index, name="market_state")
