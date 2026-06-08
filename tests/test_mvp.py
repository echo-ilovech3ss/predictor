import pytest
import pandas as pd
import numpy as np
from src.news_extractor import NewsExtractor
from run_mvp import calculate_daily_indicators

def create_dummy_daily_data(n_rows=60):
    """Generate dummy daily OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_rows, freq="1d")
    close = 100.0 + np.cumsum(np.random.normal(0.5, 2.0, n_rows))
    open_p = close - np.random.normal(0, 1.0, n_rows)
    high = np.maximum(open_p, close) + np.random.uniform(0, 2.0, n_rows)
    low = np.minimum(open_p, close) - np.random.uniform(0, 2.0, n_rows)
    volume = np.random.randint(100000, 1000000, n_rows)
    
    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)
    return df

def test_calculate_daily_indicators():
    """Verify that daily technical indicators are computed correctly with no NaNs in final rows."""
    df = create_dummy_daily_data(60)
    features = calculate_daily_indicators(df)
    
    # Check that required columns are in features
    required_cols = [
        'close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'dist_sma_20', 'dist_sma_50', 'dist_ema_20',
        'dist_bb_upper', 'dist_bb_lower', 'volume_ratio', 'daily_return'
    ]
    for col in required_cols:
        assert col in features.columns
        
    # Check no NaNs in final rows
    last_row = features.iloc[-1]
    for col in required_cols:
        assert not np.isnan(last_row[col])

def test_heuristic_news_extraction():
    """Verify the heuristic parser extracts event flags and scores matching the schema."""
    extractor = NewsExtractor()
    
    # Test positive product launch
    launch_news = "Apple unveils iPhone 16 with groundbreaking AI features"
    res1 = extractor.extract_features_heuristic(launch_news)
    assert res1['product_launch'] is True
    assert res1['sentiment'] > 0.0
    assert res1['bull_score'] > 5
    
    # Test negative lawsuit
    lawsuit_news = "DOJ sues Apple in landmark antitrust lawsuit over iPhone monopoly"
    res2 = extractor.extract_features_heuristic(lawsuit_news)
    assert res2['lawsuit'] is True
    assert res2['sentiment'] < 0.0
    assert res2['bear_score'] > 5
    assert res2['risk_score'] > 4
    
    # Check standard schema fields
    schema_fields = [
        'sentiment', 'importance', 'bull_score', 'bear_score', 'risk_score',
        'earnings', 'guidance_change', 'partnership', 'lawsuit', 'product_launch', 'management_change'
    ]
    for field in schema_fields:
        assert field in res1

def test_news_daily_aggregation():
    """Verify multiple articles on the same day aggregate correctly."""
    articles = [
        {"Date": "2024-01-01", "sentiment": 0.5, "importance": 0.8, "bull_score": 8, "bear_score": 2, "risk_score": 3, "earnings": 0, "guidance_change": 0, "partnership": 1, "lawsuit": 0, "product_launch": 0, "management_change": 0},
        {"Date": "2024-01-01", "sentiment": -0.1, "importance": 0.6, "bull_score": 4, "bear_score": 6, "risk_score": 4, "earnings": 0, "guidance_change": 0, "partnership": 0, "lawsuit": 1, "product_launch": 0, "management_change": 0}
    ]
    df = pd.DataFrame(articles)
    df['Date'] = pd.to_datetime(df['Date'])
    
    daily = df.groupby('Date').agg(
        avg_sentiment=('sentiment', 'mean'),
        max_importance=('importance', 'max'),
        bull_avg=('bull_score', 'mean'),
        bear_avg=('bear_score', 'mean'),
        risk_avg=('risk_score', 'mean'),
        partnership_count=('partnership', 'sum'),
        lawsuit_count=('lawsuit', 'sum')
    )
    
    assert daily.loc["2024-01-01", "avg_sentiment"] == pytest.approx(0.2)
    assert daily.loc["2024-01-01", "max_importance"] == 0.8
    assert daily.loc["2024-01-01", "bull_avg"] == 6.0
    assert daily.loc["2024-01-01", "bear_avg"] == 4.0
    assert daily.loc["2024-01-01", "risk_avg"] == 3.5
    assert daily.loc["2024-01-01", "partnership_count"] == 1
    assert daily.loc["2024-01-01", "lawsuit_count"] == 1
