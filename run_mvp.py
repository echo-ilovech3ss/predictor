import os
import argparse
import datetime
import pandas as pd
import numpy as np
import urllib.request
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.logger import logger
from src.news_extractor import NewsExtractor

# Create data cache directory if it doesn't exist
DATA_CACHE_DIR = "data_cache"
if not os.path.exists(DATA_CACHE_DIR):
    os.makedirs(DATA_CACHE_DIR)

NEWS_DATA_URL = "https://raw.githubusercontent.com/yannis-gerontopoulos99/Sentiment_Market_Forecasting/main/AAPL_articles.csv"
LOCAL_RAW_NEWS_PATH = os.path.join(DATA_CACHE_DIR, "aapl_news_raw.csv")
LOCAL_EXTRACTED_NEWS_PATH = os.path.join(DATA_CACHE_DIR, "aapl_news_extracted.csv")

def download_news_data() -> pd.DataFrame:
    """Download real Apple news headlines if not cached locally."""
    if not os.path.exists(LOCAL_RAW_NEWS_PATH):
        logger.info(f"Downloading real stock news from {NEWS_DATA_URL}...")
        try:
            urllib.request.urlretrieve(NEWS_DATA_URL, LOCAL_RAW_NEWS_PATH)
            logger.info(f"Saved raw news to {LOCAL_RAW_NEWS_PATH}")
        except Exception as e:
            logger.error(f"Failed to download news dataset: {e}")
            raise e
            
    df = pd.read_csv(LOCAL_RAW_NEWS_PATH)
    logger.info(f"Loaded {len(df)} real news articles from local cache.")
    return df

def fetch_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily stock price data from Yahoo Finance."""
    import yfinance as yf
    logger.info(f"Fetching daily market data for {symbol} from {start_date} to {end_date}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    
    if df.empty:
        raise ValueError(f"No market data found for symbol {symbol}")
        
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })
    logger.info(f"Fetched {len(df)} trading days for {symbol}.")
    return df

def calculate_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the required technical indicators for the daily candles."""
    data = df.copy()
    
    # 1. RSI (14)
    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    # Avoid divide by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['rsi'] = 100 - (100 / (1 + rs))
    data['rsi'] = data['rsi'].fillna(50)
    
    # 2. MACD (12, 26, 9)
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # 3. SMA 20, SMA 50
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    
    # 4. EMA 20
    data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
    
    # 5. Bollinger Bands (20, 2)
    bb_middle = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = bb_middle + 2 * bb_std
    data['bb_lower'] = bb_middle - 2 * bb_std
    
    # 6. Volume Ratio (current / 20-day average)
    vol_ma_20 = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / vol_ma_20.replace(0, 1.0)
    
    # Create final technical features
    features = pd.DataFrame(index=data.index)
    features['close'] = data['close']
    features['volume'] = data['volume']
    features['rsi'] = data['rsi']
    features['macd'] = data['macd']
    features['macd_signal'] = data['macd_signal']
    features['macd_hist'] = data['macd_hist']
    
    # Normalized features to prevent scale leaks
    features['dist_sma_20'] = (data['close'] - data['sma_20']) / data['sma_20']
    features['dist_sma_50'] = (data['close'] - data['sma_50']) / data['sma_50']
    features['dist_ema_20'] = (data['close'] - data['ema_20']) / data['ema_20']
    
    safe_close = data['close'].replace(0, 1.0)
    features['dist_bb_upper'] = (data['bb_upper'] - data['close']) / safe_close
    features['dist_bb_lower'] = (data['close'] - data['bb_lower']) / safe_close
    features['volume_ratio'] = data['volume_ratio']
    features['daily_return'] = data['close'].pct_change().fillna(0)
    
    # Fill any remaining NaNs with 0
    features = features.fillna(0)
    return features

def main():
    parser = argparse.ArgumentParser(description="Stock AI MVP Pipeline using real data and LLM")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock ticker to run MVP for (default: AAPL)")
    parser.add_argument("--sample-size", type=int, default=15, help="Number of articles to run LLM extraction on for demo")
    parser.add_argument("--use-llm-all", action="store_true", help="If set, calls the LLM for all articles (takes longer)")
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    logger.info(f"=== Running Stock AI MVP for {symbol} ===")
    
    # ----------------------------------------------------
    # Phase 1: Data Collection
    # ----------------------------------------------------
    logger.info("--- Phase 1: Data Collection ---")
    news_df = download_news_data()
    
    # Convert Dates and find range
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    min_date = news_df['Date'].min().strftime("%Y-%m-%d")
    max_date = news_df['Date'].max().strftime("%Y-%m-%d")
    logger.info(f"News dates range from {min_date} to {max_date}")
    
    # Add buffer to market dates to accommodate indicator windows
    start_date_market = (news_df['Date'].min() - datetime.timedelta(days=100)).strftime("%Y-%m-%d")
    end_date_market = (news_df['Date'].max() + datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    
    raw_market_df = fetch_market_data(symbol, start_date_market, end_date_market)
    
    # Calculate indicators
    market_features = calculate_daily_indicators(raw_market_df)
    
    # ----------------------------------------------------
    # Phase 2: LLM Feature Extraction
    # ----------------------------------------------------
    logger.info("--- Phase 2: LLM Feature Extraction ---")
    extractor = NewsExtractor()
    
    # Let's check if we have a locally cached processed news dataset
    processed_news_list = []
    
    if os.path.exists(LOCAL_EXTRACTED_NEWS_PATH) and not args.use_llm_all:
        logger.info(f"Loading cached extracted news features from {LOCAL_EXTRACTED_NEWS_PATH}")
        extracted_df = pd.read_csv(LOCAL_EXTRACTED_NEWS_PATH)
        extracted_df['Date'] = pd.to_datetime(extracted_df['Date'])
    else:
        # We need to run extraction
        logger.info(f"Running extraction on {len(news_df)} headlines...")
        
        # 1. Demo LLM Extraction on sample size
        demo_count = min(args.sample_size, len(news_df))
        logger.info(f"Demonstrating actual DeepSeek-v4-flash-free LLM extraction on {demo_count} sample articles:")
        
        for idx in range(demo_count):
            row = news_df.iloc[idx]
            title = row['Title']
            date = row['Date']
            
            logger.info(f"[{idx+1}/{demo_count}] Querying DeepSeek for: '{title}'...")
            try:
                features = extractor.extract_features_llm(title)
                features['Title'] = title
                features['Date'] = date
                processed_news_list.append(features)
                logger.info(f"  Result Sentiment: {features['sentiment']}, Bull: {features['bull_score']}, Bear: {features['bear_score']}, Events: {[k for k,v in features.items() if v is True]}")
            except Exception as e:
                logger.error(f"  LLM Query failed, falling back to heuristic: {e}")
                features = extractor.extract_features_heuristic(title)
                features['Title'] = title
                features['Date'] = date
                processed_news_list.append(features)
                
        # 2. Extract features for the rest
        if args.use_llm_all:
            logger.info("Running LLM extraction on remaining articles (this may take a while)...")
            for idx in range(demo_count, len(news_df)):
                row = news_df.iloc[idx]
                title = row['Title']
                date = row['Date']
                if idx % 50 == 0:
                    logger.info(f"Progress: {idx}/{len(news_df)}")
                try:
                    features = extractor.extract_features_llm(title)
                except Exception:
                    features = extractor.extract_features_heuristic(title)
                features['Title'] = title
                features['Date'] = date
                processed_news_list.append(features)
        else:
            logger.info("Processing remaining articles with the heuristic rule-based parser...")
            for idx in range(demo_count, len(news_df)):
                row = news_df.iloc[idx]
                title = row['Title']
                date = row['Date']
                features = extractor.extract_features_heuristic(title)
                features['Title'] = title
                features['Date'] = date
                processed_news_list.append(features)
                
        extracted_df = pd.DataFrame(processed_news_list)
        extracted_df.to_csv(LOCAL_EXTRACTED_NEWS_PATH, index=False)
        logger.info(f"Saved extracted news features to {LOCAL_EXTRACTED_NEWS_PATH}")
        
    # ----------------------------------------------------
    # Phase 3: Daily Aggregation
    # ----------------------------------------------------
    logger.info("--- Phase 3: Daily Aggregation ---")
    
    # Convert booleans to ints for sum aggregations
    bool_cols = ["earnings", "guidance_change", "partnership", "lawsuit", "product_launch", "management_change"]
    for col in bool_cols:
        extracted_df[col] = extracted_df[col].astype(int)
        
    extracted_df['weighted_sentiment'] = extracted_df['sentiment'] * extracted_df['importance']
        
    daily_news = extracted_df.groupby('Date').agg(
        avg_sentiment=('sentiment', 'mean'),
        weighted_sentiment=('weighted_sentiment', 'mean'),
        max_importance=('importance', 'max'),
        bull_avg=('bull_score', 'mean'),
        bear_avg=('bear_score', 'mean'),
        risk_avg=('risk_score', 'mean'),
        partnership_count=('partnership', 'sum'),
        lawsuit_count=('lawsuit', 'sum'),
        earnings_count=('earnings', 'sum'),
        guidance_count=('guidance_change', 'sum'),
        product_launch_count=('product_launch', 'sum'),
        management_change_count=('management_change', 'sum')
    )
    logger.info(f"Aggregated news into {len(daily_news)} distinct daily records.")
    
    # ----------------------------------------------------
    # Phase 4: Dataset Creation (Merge Market + News)
    # ----------------------------------------------------
    logger.info("--- Phase 4: Dataset Creation ---")
    
    # Align indexes as localized/timezone-naive dates
    market_features.index = market_features.index.tz_localize(None).normalize()
    daily_news.index = daily_news.index.tz_localize(None).normalize()
    
    # Combine market features and news features
    dataset = market_features.join(daily_news, how='left')
    
    # Fill dates with NO news with neutral values
    dataset['avg_sentiment'] = dataset['avg_sentiment'].fillna(0.0)
    dataset['weighted_sentiment'] = dataset['weighted_sentiment'].fillna(0.0)
    dataset['max_importance'] = dataset['max_importance'].fillna(0.0)
    dataset['bull_avg'] = dataset['bull_avg'].fillna(5.0)
    dataset['bear_avg'] = dataset['bear_avg'].fillna(5.0)
    dataset['risk_avg'] = dataset['risk_avg'].fillna(2.0)
    
    count_cols = [
        "partnership_count", "lawsuit_count", "earnings_count", 
        "guidance_count", "product_launch_count", "management_change_count"
    ]
    for col in count_cols:
        dataset[col] = dataset[col].fillna(0.0)
        
    # Add Lagged and Rolling Features to prevent overfitting and capture trend momentum
    dataset['avg_sentiment_roll3'] = dataset['avg_sentiment'].rolling(window=3).mean().fillna(0)
    dataset['avg_sentiment_lag1'] = dataset['avg_sentiment'].shift(1).fillna(0)
    dataset['daily_return_lag1'] = dataset['daily_return'].shift(1).fillna(0)
    dataset['weighted_sentiment_lag1'] = dataset['weighted_sentiment'].shift(1).fillna(0)
        
    # ----------------------------------------------------
    # Phase 5: Target Variable Definition (5-day future return direction)
    # ----------------------------------------------------
    logger.info("--- Phase 5: Target Variable ---")
    # future_return = Close(t+5) - Close(t)
    # Target = 1 if future_return > 0 else 0
    # Note: shift(-5) on daily data shifts future close to today's row (computed on full market data to avoid NaNs at the end of the news range)
    dataset['future_close'] = dataset['close'].shift(-5)
    dataset['future_return'] = dataset['future_close'] - dataset['close']
    dataset['target'] = (dataset['future_return'] > 0).astype(int)
    
    # Slice the dataset to start precisely when the news data starts
    news_start_date = daily_news.index.min()
    news_end_date = daily_news.index.max()
    dataset = dataset.loc[news_start_date:news_end_date]
    logger.info(f"Combined dataset spans {len(dataset)} trading days (from {dataset.index.min().date()} to {dataset.index.max().date()}).")
    
    # Drop the last 5 rows if they don't have future returns (which will be at the end of our fetched market data range)
    cleaned_dataset = dataset.dropna(subset=['future_close'])
    logger.info(f"Cleaned dataset for ML contains {len(cleaned_dataset)} rows after lookahead drop.")
    
    # ----------------------------------------------------
    # Phase 6: Training
    # ----------------------------------------------------
    logger.info("--- Phase 6: Training ---")
    
    technical_cols = [
        'rsi', 'macd', 'macd_signal', 'macd_hist', 
        'dist_sma_20', 'dist_sma_50', 'dist_ema_20', 
        'dist_bb_upper', 'dist_bb_lower', 'volume_ratio', 'daily_return',
        'daily_return_lag1'
    ]
    
    news_cols = [
        'avg_sentiment', 'weighted_sentiment', 'max_importance', 'bull_avg', 'bear_avg', 'risk_avg',
        'partnership_count', 'lawsuit_count', 'earnings_count',
        'guidance_count', 'product_launch_count', 'management_change_count',
        'avg_sentiment_roll3', 'avg_sentiment_lag1', 'weighted_sentiment_lag1'
    ]
    
    # Split by dates: Train 2023-2024, Test 2025
    train_split_end = pd.Timestamp("2024-12-31")
    
    train_data = cleaned_dataset.loc[:train_split_end]
    test_data = cleaned_dataset.loc[train_split_end + pd.Timedelta(days=1):]
    
    logger.info(f"Train split (2023-2024): {len(train_data)} trading days")
    logger.info(f"Test split (2025): {len(test_data)} trading days")
    
    if len(test_data) == 0:
        logger.error("Test dataset is empty. Check dates!")
        return
        
    X_train_A = train_data[technical_cols]
    y_train = train_data['target']
    X_test_A = test_data[technical_cols]
    y_test = test_data['target']
    
    X_train_B = train_data[technical_cols + news_cols]
    X_test_B = test_data[technical_cols + news_cols]
    
    # Optimized XGBClassifier parameters to prevent overfitting on smaller sample
    params = {
        'max_depth': 3,
        'n_estimators': 300,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 2.0,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    # Model A: Technical Indicators Only
    logger.info("Training Model A (Technical Indicators Only)...")
    model_A = XGBClassifier(**params)
    model_A.fit(X_train_A, y_train)
    
    # Model B: Technical Indicators + News Features
    logger.info("Training Model B (Technical Indicators + News Features)...")
    model_B = XGBClassifier(**params)
    model_B.fit(X_train_B, y_train)
    
    # ----------------------------------------------------
    # Phase 7: Evaluation
    # ----------------------------------------------------
    logger.info("--- Phase 7: Evaluation ---")
    
    def evaluate(model, X_test, y_test):
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, probas)
        cm = confusion_matrix(y_test, preds)
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }
        
    results_A = evaluate(model_A, X_test_A, y_test)
    results_B = evaluate(model_B, X_test_B, y_test)
    
    # Print Comparison Table
    print("\n========================================================")
    print("                STOCK AI MVP EVALUATION REPORT          ")
    print("========================================================")
    print(f"Target Symbol:       {symbol} (Daily)")
    print(f"Train Period:        {train_data.index.min().date()} to {train_data.index.max().date()}")
    print(f"Test Period:         {test_data.index.min().date()} to {test_data.index.max().date()}")
    print(f"Target:              1 if Price rises over next 5 trading days, else 0")
    print("--------------------------------------------------------")
    print(f"{'Metric':<20} | {'Model A (Tech Only)':<20} | {'Model B (Tech+News)':<20}")
    print(f"{'-'*20}-+-{'-'*20}-+-{'-'*20}")
    
    print(f"{'Accuracy':<20} | {results_A['accuracy']:<20.4f} | {results_B['accuracy']:<20.4f}")
    print(f"{'Precision':<20} | {results_A['precision']:<20.4f} | {results_B['precision']:<20.4f}")
    print(f"{'Recall':<20} | {results_A['recall']:<20.4f} | {results_B['recall']:<20.4f}")
    print(f"{'F1 Score':<20} | {results_A['f1']:<20.4f} | {results_B['f1']:<20.4f}")
    print(f"{'ROC AUC':<20} | {results_A['roc_auc']:<20.4f} | {results_B['roc_auc']:<20.4f}")
    print("--------------------------------------------------------")
    
    improvement = results_B['accuracy'] - results_A['accuracy']
    improvement_pct = improvement * 100
    
    print(f"Accuracy Improvement (Model B - Model A): {improvement_pct:+.2f}%")
    print("--------------------------------------------------------")
    print("Confusion Matrix - Model A (Tech Only):")
    print(results_A['confusion_matrix'])
    print("Confusion Matrix - Model B (Tech + News):")
    print(results_B['confusion_matrix'])
    print("========================================================")
    
    # Save a markdown report (walkthrough) to artifacts
    report_content = f"""# Stock AI MVP Walkthrough Report

This MVP tests whether incorporating LLM-extracted news sentiment and events improves prediction accuracy of daily stock price direction 5 trading days ahead.

## Parameters & Setup
- **Target Stock**: {symbol} (Apple Inc.)
- **Market Data**: Daily OHLCV with Technical Indicators (RSI, MACD, SMAs, EMAs, Bollinger Bands, Volume Ratio)
- **News Data**: Real {symbol} news headlines dataset (`AAPL_articles.csv` containing {len(news_df)} rows)
- **Extraction Model**: DeepSeek v4 Flash via OpenCode Zen API (with heuristic processing for bulk history)
- **ML Classifier**: XGBClassifier (max_depth=6, n_estimators=500, learning_rate=0.05)
- **Evaluation splits**: Train (2023–2024), Test (2025)

## Performance Summary

| Metric | Model A (Technical Only) | Model B (Technical + News) | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | {results_A['accuracy']:.4f} | {results_B['accuracy']:.4f} | {improvement_pct:+.2f}% |
| **Precision** | {results_A['precision']:.4f} | {results_B['precision']:.4f} | {(results_B['precision'] - results_A['precision'])*100:+.2f}% |
| **Recall** | {results_A['recall']:.4f} | {results_B['recall']:.4f} | {(results_B['recall'] - results_A['recall'])*100:+.2f}% |
| **F1 Score** | {results_A['f1']:.4f} | {results_B['f1']:.4f} | {(results_B['f1'] - results_A['f1'])*100:+.2f}% |
| **ROC AUC** | {results_A['roc_auc']:.4f} | {results_B['roc_auc']:.4f} | {(results_B['roc_auc'] - results_A['roc_auc'])*100:+.2f}% |

### Confusion Matrix (Model A)
```
{results_A['confusion_matrix']}
```

### Confusion Matrix (Model B)
```
{results_B['confusion_matrix']}
```

## Answer to Core Question
> **Does adding news understanding improve prediction accuracy over pure market data?**

{"**YES**" if improvement > 0 else "**NO**" or "**NO CHANGE**" if improvement == 0 else "**NO (slight degradation)**"}. The news sentiment features resulted in an accuracy shift of **{improvement_pct:+.2f}%** on the out-of-sample 2025 test set.

"""
    
    # Write walkthrough report
    walkthrough_path = "/Users/arunmehta/.gemini/antigravity/brain/f7b90d3f-b23d-4ea0-8898-305d806e2758/walkthrough.md"
    with open(walkthrough_path, "w") as f:
        f.write(report_content)
    logger.info(f"Walkthrough report written to {walkthrough_path}")

if __name__ == "__main__":
    main()
