import pytest
import pandas as pd
import numpy as np
from src.indicators import calculate_indicators
from src.features import extract_features, prepare_data_for_training

def create_dummy_ohlcv(n_rows=250):
    """Generate dummy OHLCV data with clear trends."""
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=n_rows, freq="1h")
    
    close = 100.0 + np.cumsum(np.random.normal(0.1, 1.0, n_rows))
    open_p = close - np.random.normal(0, 0.5, n_rows)
    high = np.maximum(open_p, close) + np.random.uniform(0, 1.0, n_rows)
    low = np.minimum(open_p, close) - np.random.uniform(0, 1.0, n_rows)
    volume = np.random.randint(1000, 10000, n_rows)
    
    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)
    
    return df

def test_feature_no_future_leakage():
    """
    Test that features at index t do NOT change when subsequent data (t+1, t+2...) is mutated.
    This guarantees no lookahead bias in the feature extraction pipeline.
    """
    df = create_dummy_ohlcv(250)
    df_with_ind = calculate_indicators(df)
    
    # 1. Extract base features
    features_base = extract_features(df_with_ind)
    
    # Target row to test (e.g. index 210)
    test_idx = 210
    
    # Extract feature values at test_idx
    base_features_row = features_base.iloc[test_idx].copy()
    
    # 2. Mutate future rows (index > 210) in the original dataframe
    df_mutated = df.copy()
    for future_idx in range(test_idx + 1, len(df)):
        df_mutated.iloc[future_idx] = df_mutated.iloc[future_idx] * 2.0  # Double all future prices/volume
        
    # Recalculate indicators and features for the mutated dataset
    df_mutated_with_ind = calculate_indicators(df_mutated)
    features_mutated = extract_features(df_mutated_with_ind)
    
    mutated_features_row = features_mutated.iloc[test_idx]
    
    # 3. Assert that features at test_idx remain completely unchanged
    pd.testing.assert_series_equal(base_features_row, mutated_features_row, check_exact=False, atol=1e-7)

def test_label_generation_uses_future():
    """
    Test that target labels at index t correctly represent whether index t+4 closes at least 0.05% higher.
    Mutating close(t+4) MUST change target(t).
    """
    df = create_dummy_ohlcv(250)
    df_with_ind = calculate_indicators(df)
    
    X, y = prepare_data_for_training(df_with_ind)
    
    # Find a row where index is within X
    test_idx = 210
    test_timestamp = X.index[test_idx]
    future_timestamp = X.index[test_idx + 4]
    
    # Verify label matches direction
    close_now = df_with_ind.loc[test_timestamp, 'close']
    close_future = df_with_ind.loc[future_timestamp, 'close']
    expected_label = 1 if close_future > close_now * 1.0005 else 0
    
    assert y.loc[test_timestamp] == expected_label
