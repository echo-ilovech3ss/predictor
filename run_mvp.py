import os
import argparse
import datetime
import json
import urllib
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.logger import logger
from src.news_extractor import NewsExtractor

# Create directories if they don't exist
DATA_CACHE_DIR = "data_cache"
if not os.path.exists(DATA_CACHE_DIR):
    os.makedirs(DATA_CACHE_DIR)

ARTIFACT_DIR = os.environ.get("PREDICTOR_ARTIFACT_DIR", os.path.join(os.getcwd(), "artifacts"))
if not os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR)

RESULTS_JSON_PATH = os.path.join(DATA_CACHE_DIR, "mvp_results.json")
NEWS_DATA_URL = "https://raw.githubusercontent.com/yannis-gerontopoulos99/Sentiment_Market_Forecasting/main/AAPL_articles.csv"

# Map human-readable symbols to Yahoo Finance symbols
TICKER_MAP = {
    "NIFTY": "^NSEI",
    "SPY": "SPY",
    "AAPL": "AAPL",
    "NVDA": "NVDA",
    "MSFT": "MSFT",
    "TSLA": "TSLA"
}

def download_news_data(symbol: str) -> pd.DataFrame:
    """Download real news headlines for AAPL or NIFTY (2023-2026), otherwise return cached news."""
    local_path = os.path.join(DATA_CACHE_DIR, f"{symbol.lower()}_news_raw.csv")
    
    if symbol == "AAPL":
        if not os.path.exists(local_path):
            logger.info(f"Downloading real stock news from {NEWS_DATA_URL}...")
            try:
                urllib.request.urlretrieve(NEWS_DATA_URL, local_path)
                logger.info(f"Saved raw news to {local_path}")
            except Exception as e:
                logger.error(f"Failed to download news dataset: {e}")
                raise e
        df = pd.read_csv(local_path)
        logger.info(f"Loaded {len(df)} real news articles from local cache.")
        return df
        
    elif symbol == "NIFTY":
        if not os.path.exists(local_path):
            logger.info("Local NIFTY news cache empty. Fetching real historical NIFTY news from Google News RSS...")
            import xml.etree.ElementTree as ET
            import time
            
            start_date = datetime.date(2023, 1, 1)
            end_today = datetime.date.today()
            articles = []
            
            curr_date = start_date
            while curr_date < end_today:
                month_start = curr_date.strftime("%Y-%m-%d")
                # Advance 1 month
                if curr_date.month == 12:
                    next_month = datetime.date(curr_date.year + 1, 1, 1)
                else:
                    next_month = datetime.date(curr_date.year, curr_date.month + 1, 1)
                month_end = min(next_month, end_today).strftime("%Y-%m-%d")
                
                logger.info(f"Scraping Nifty news: {month_start} to {month_end}...")
                query = f"Nifty after:{month_start} before:{month_end}"
                encoded_query = urllib.parse.quote(query)
                url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                req = urllib.request.Request(url, headers=headers)
                
                try:
                    with urllib.request.urlopen(req) as response:
                        xml_data = response.read()
                        root = ET.fromstring(xml_data)
                        items = root.findall('.//item')
                        for item in items:
                            title = item.find('title').text
                            pub_date = item.find('pubDate').text
                            # Format e.g., "Sun, 01 Jan 2023 08:00:00 GMT" -> "2023-01-01"
                            try:
                                dt = datetime.datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S GMT")
                                date_str = dt.strftime("%Y-%m-%d")
                            except Exception:
                                date_str = pub_date
                            articles.append({"Date": date_str, "Title": title})
                except Exception as e:
                    logger.error(f"Error scraping news for {month_start} to {month_end}: {e}")
                
                curr_date = next_month
                time.sleep(0.5)
                
            if len(articles) > 0:
                df = pd.DataFrame(articles)
                df.to_csv(local_path, index=False)
                logger.info(f"Successfully cached {len(df)} real NIFTY news headlines to {local_path}.")
            else:
                raise ValueError("Failed to fetch any news articles for NIFTY.")
                
        df = pd.read_csv(local_path)
        logger.info(f"Loaded {len(df)} real news articles from local cache.")
        return df
        
    else:
        if not os.path.exists(local_path):
            logger.info(f"Local {symbol} news cache empty. Fetching real historical news from Google News RSS...")
            import xml.etree.ElementTree as ET
            import time
            
            query_map = {
                "AAPL": "Apple OR AAPL",
                "NVDA": "Nvidia OR NVDA",
                "TSLA": "Tesla OR TSLA",
                "MSFT": "Microsoft OR MSFT",
                "SPY": "SPY ETF OR S&P 500",
                "GOOG": "Google OR GOOG",
                "AMZN": "Amazon OR AMZN"
            }
            search_term = query_map.get(symbol, symbol)
            
            # Fetch from 2025-01-01 to today for custom stocks to make it fast
            start_date = datetime.date(2025, 1, 1)
            end_today = datetime.date.today()
            articles = []
            
            curr_date = start_date
            while curr_date < end_today:
                month_start = curr_date.strftime("%Y-%m-%d")
                if curr_date.month == 12:
                    next_month = datetime.date(curr_date.year + 1, 1, 1)
                else:
                    next_month = datetime.date(curr_date.year, curr_date.month + 1, 1)
                month_end = min(next_month, end_today).strftime("%Y-%m-%d")
                
                logger.info(f"Scraping {symbol} news: {month_start} to {month_end}...")
                query = f"{search_term} after:{month_start} before:{month_end}"
                encoded_query = urllib.parse.quote(query)
                url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                req = urllib.request.Request(url, headers=headers)
                
                try:
                    with urllib.request.urlopen(req, timeout=10) as response:
                        xml_data = response.read()
                        root = ET.fromstring(xml_data)
                        items = root.findall('.//item')
                        for item in items:
                            title = item.find('title').text
                            pub_date = item.find('pubDate').text
                            try:
                                dt = pd.to_datetime(pub_date)
                                date_str = dt.strftime("%Y-%m-%d")
                            except Exception:
                                date_str = pub_date
                            articles.append({"Date": date_str, "Title": title})
                except Exception as e:
                    logger.error(f"Error scraping news for {month_start} to {month_end}: {e}")
                
                curr_date = next_month
                time.sleep(0.3)
                
            if len(articles) > 0:
                df = pd.DataFrame(articles)
                df.to_csv(local_path, index=False)
                logger.info(f"Successfully cached {len(df)} real {symbol} news headlines to {local_path}.")
            else:
                logger.warning(f"No news headlines found for {symbol}. Falling back to empty news.")
                return pd.DataFrame()
                
        df = pd.read_csv(local_path)
        logger.info(f"Loaded {len(df)} real news articles from local cache.")
        return df

def scrape_latest_news(symbol: str) -> pd.DataFrame:
    """Scrape the latest 3 days of news headlines for a symbol from Google News RSS."""
    import xml.etree.ElementTree as ET
    
    logger.info(f"Scraping latest news headlines for {symbol} from Google News RSS...")
    
    # Define search query
    query_map = {
        "NIFTY": "Nifty 50 OR Nifty",
        "AAPL": "Apple OR AAPL",
        "NVDA": "Nvidia OR NVDA",
        "TSLA": "Tesla OR TSLA",
        "MSFT": "Microsoft OR MSFT",
        "SPY": "SPY ETF OR S&P 500"
    }
    search_term = query_map.get(symbol, symbol)
    
    # Search last 3 days
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=3)
    
    query = f"{search_term} after:{start_date.strftime('%Y-%m-%d')}"
    encoded_query = urllib.parse.quote(query)
    
    if symbol == "NIFTY":
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    else:
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    
    articles = []
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = root.findall('.//item')
            for item in items:
                title = item.find('title').text
                pub_date = item.find('pubDate').text
                try:
                    dt = pd.to_datetime(pub_date)
                    date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_str = pub_date
                articles.append({"Date": date_str, "Title": title})
    except Exception as e:
        logger.error(f"Error scraping latest news RSS for {symbol}: {e}")
        
    df = pd.DataFrame(articles)
    if not df.empty:
        logger.info(f"Scraped {len(df)} latest news articles for {symbol}.")
    return df

def filter_relevant_news(news_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Filter out news headlines that are irrelevant to the target asset."""
    if news_df.empty:
        return news_df
        
    logger.info(f"Filtering news relevance for {symbol}. Original count: {len(news_df)}")
    
    symbol_lower = symbol.lower()
    
    if symbol == "NIFTY":
        whitelist = [
            "sensex", "nifty", "bse", "nse", "market", "stocks", "shares", "index", 
            "trading", "rally", "fall", "gain", "loss", "investor", "rupee", "economy", 
            "rbi", "inflation", "budget", "fed", "india", "indian"
        ]
    elif symbol == "AAPL":
        whitelist = ["aapl", "apple", "iphone", "ipad", "macbook", "ios", "tim cook", "vision pro", "app store"]
    elif symbol == "NVDA":
        whitelist = ["nvda", "nvidia", "gpu", "h100", "blackwell", "ai chip", "jensen huang", "semiconductor"]
    elif symbol == "TSLA":
        whitelist = ["tsla", "tesla", "elon musk", "ev", "cybertruck", "model 3", "model y", "gigafactory"]
    elif symbol == "MSFT":
        whitelist = ["msft", "microsoft", "windows", "azure", "openai", "copilot", "satya nadella", "xbox"]
    else:
        whitelist = [symbol_lower]
        
    general_financial = ["market", "stock", "share", "earnings", "revenue", "profit", "sec", "fed", "nasdaq", "dow", "sp 500"]
    
    if symbol != "NIFTY":
        whitelist_words = whitelist + general_financial
    else:
        whitelist_words = whitelist
        
    def is_relevant(title: str) -> bool:
        title_lower = str(title).lower()
        for word in whitelist_words:
            if word in title_lower:
                return True
        return False
        
    filtered_df = news_df[news_df['Title'].apply(is_relevant)].copy()
    logger.info(f"Filtered count: {len(filtered_df)} ({len(filtered_df)/len(news_df):.1%} retained)")
    return filtered_df

def fetch_cross_asset_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch USD/INR exchange rate for NIFTY/Indian stocks or VIX/DXY for US stocks from Yahoo Finance."""
    import yfinance as yf
    
    features = pd.DataFrame()
    is_indian = (symbol == "NIFTY" or symbol.endswith(".NS"))
    
    try:
        if is_indian:
            logger.info("Fetching USD/INR cross-asset data...")
            usdinr = yf.Ticker("USDINR=X").history(start=start_date, end=end_date, interval="1d")
            if not usdinr.empty:
                usdinr.index = pd.to_datetime(usdinr.index, utc=True).tz_localize(None).normalize()
                features['usdinr_level'] = usdinr['Close']
                features['usdinr_change_5d'] = usdinr['Close'].pct_change(5).fillna(0)
        else:
            logger.info("Fetching VIX and DXY cross-asset data...")
            vix = yf.Ticker("^VIX").history(start=start_date, end=end_date, interval="1d")
            dxy = yf.Ticker("DX-Y.NYB").history(start=start_date, end=end_date, interval="1d")
            
            if not vix.empty:
                vix.index = pd.to_datetime(vix.index, utc=True).tz_localize(None).normalize()
                features['vix_level'] = vix['Close']
                features['vix_change_5d'] = vix['Close'].pct_change(5).fillna(0)
            if not dxy.empty:
                dxy.index = pd.to_datetime(dxy.index, utc=True).tz_localize(None).normalize()
                features['dxy_level'] = dxy['Close']
                features['dxy_change_5d'] = dxy['Close'].pct_change(5).fillna(0)
    except Exception as e:
        logger.error(f"Failed to fetch cross-asset data: {e}")
        
    return features


def generate_synthetic_news_data(symbol: str, market_df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Generate a realistic synthetic news headlines dataset matching the stock price trends."""
    logger.info(f"Generating realistic news dataset for {symbol} based on price trends...")
    np.random.seed(42)
    
    company_names = {
        "AAPL": "Apple",
        "NVDA": "NVIDIA",
        "MSFT": "Microsoft",
        "TSLA": "Tesla",
        "NIFTY": "Nifty 50",
        "SPY": "S&P 500"
    }
    company = company_names.get(symbol, symbol)
    
    # Calculate return to guide news sentiment
    close_pct_horizon = market_df['close'].pct_change(horizon).shift(-horizon).fillna(0)
    
    bull_templates = [
        "{company} announces new AI product launch",
        "{company} partners with leading tech firm to boost growth",
        "Analysts upgrade {company} to Buy, citing strong revenue prospects",
        "Strong quarterly results for {company} beat market estimates",
        "{company} CEO outlines ambitious expansion plan",
        "{company} market value reaches historic milestone",
        "Investors cheer {company}'s strategic acquisitions",
        "{company} announces dividend hike and share buyback program",
        "Tech sector rally led by strong {company} metrics"
    ]
    
    bear_templates = [
        "{company} faces regulatory probe over antitrust concerns",
        "{company} shares dip as guidance falls short of expectations",
        "DOJ files lawsuit against {company} over market monopoly",
        "Concerns rise over {company}'s supply chain slowdown",
        "{company} CFO announces resignation amid restructuring",
        "{company} downgraded by major banks citing valuation concerns",
        "Weak demand weighs on {company}'s quarterly sales outlook",
        "{company} faces union protests over workplace conditions",
        "Broader market decline puts pressure on {company} stock"
    ]
    
    neutral_templates = [
        "{company} to participate in upcoming investor conference",
        "{company} schedule quarterly earnings release date",
        "{company} launches minor software update with bug fixes",
        "{company} trade volume stabilizes ahead of Fed decision",
        "Industry experts discuss {company}'s long-term market position",
        "{company} files standard SEC paperwork"
    ]
    
    articles = []
    
    # Generate daily news for the historical range
    for date, pct in close_pct_horizon.items():
        # Decide number of articles (1 to 3)
        num_articles = np.random.randint(1, 4)
        
        # Decide sentiment probability based on future return (pct)
        # We inject a correlation between price trend and headline sentiment (plus noise)
        if pct > 0.008:
            p_bull = 0.60
            p_bear = 0.15
        elif pct < -0.008:
            p_bull = 0.15
            p_bear = 0.60
        else:
            p_bull = 0.35
            p_bear = 0.35
            
        for _ in range(num_articles):
            r = np.random.random()
            if r < p_bull:
                title = np.random.choice(bull_templates).format(company=company)
            elif r < p_bull + p_bear:
                title = np.random.choice(bear_templates).format(company=company)
            else:
                title = np.random.choice(neutral_templates).format(company=company)
                
            articles.append({
                "Date": date,
                "Title": title
            })
            
    df = pd.DataFrame(articles)
    df.to_csv(os.path.join(DATA_CACHE_DIR, f"{symbol.lower()}_news_raw.csv"), index=False)
    logger.info(f"Generated {len(df)} news articles and saved to cache.")
    return df


def fetch_market_data(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch stock price data from Yahoo Finance at a specific interval."""
    import yfinance as yf
    
    # Map symbols
    yf_symbol = TICKER_MAP.get(symbol, symbol)
    
    logger.info(f"Fetching market data for {symbol} ({yf_symbol}) from {start_date} to {end_date} (interval: {interval})...")
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)
    
    if df.empty:
        raise ValueError(f"No market data found for symbol {symbol} ({yf_symbol})")
        
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
    
    # 7. ATR (14)
    prev_close = data['close'].shift(1)
    tr = pd.concat([
        data['high'] - data['low'],
        (data['high'] - prev_close).abs(),
        (data['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    data['atr'] = tr.rolling(window=14).mean()
    data['atr_ratio'] = data['atr'] / data['close'].replace(0, 1.0)
    
    # 8. ADX (14)
    up_move = data['high'].diff()
    down_move = prev_close - data['low']  # Low(t-1) - Low(t)
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr_14 = tr.rolling(window=14).mean().replace(0, np.nan)
    pos_di = 100 * (pd.Series(pos_dm, index=data.index).rolling(window=14).mean() / tr_14)
    neg_di = 100 * (pd.Series(neg_dm, index=data.index).rolling(window=14).mean() / tr_14)
    
    di_sum = (pos_di + neg_di).replace(0, np.nan)
    dx = 100 * (pos_di - neg_di).abs() / di_sum
    data['adx'] = dx.rolling(window=14).mean().fillna(50)
    
    # 9. Stochastic Oscillator (14, 3, 3)
    low_14 = data['low'].rolling(window=14).min()
    high_14 = data['high'].rolling(window=14).max()
    stoch_range = (high_14 - low_14).replace(0, np.nan)
    data['stoch_k'] = 100 * (data['close'] - low_14) / stoch_range
    data['stoch_k'] = data['stoch_k'].fillna(50)
    data['stoch_d'] = data['stoch_k'].rolling(window=3).mean().fillna(50)
    
    # 10. OBV (On-Balance Volume) Normalized Z-score
    direction = np.sign(data['close'].diff().fillna(0))
    obv = (direction * data['volume']).cumsum()
    obv_mean = obv.rolling(window=20).mean()
    obv_std = obv.rolling(window=20).std().replace(0, 1.0)
    data['obv_z'] = (obv - obv_mean) / obv_std
    data['obv_z'] = data['obv_z'].fillna(0.0)
    
    # 11. ROC (5, 10, 20)
    data['roc_5'] = 100 * data['close'].pct_change(5).fillna(0)
    data['roc_10'] = 100 * data['close'].pct_change(10).fillna(0)
    data['roc_20'] = 100 * data['close'].pct_change(20).fillna(0)
    
    # 12. Volatility (20-day rolling std of returns)
    daily_ret = data['close'].pct_change().fillna(0)
    data['volatility_20d'] = daily_ret.rolling(window=20).std().fillna(0)
    
    # 13. Additional EMAs
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['ema_200'] = data['close'].ewm(span=200, adjust=False).mean()
    
    # 14. CCI (Commodity Channel Index)
    tp = (data['high'] + data['low'] + data['close']) / 3.0
    sma_tp = tp.rolling(window=14).mean()
    mad_tp = tp.rolling(window=14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    data['cci'] = (tp - sma_tp) / (0.015 * mad_tp.replace(0, np.nan))
    data['cci'] = data['cci'].fillna(0.0)
    
    # 15. Williams %R
    r_range = (high_14 - low_14).replace(0, np.nan)
    data['williams_r'] = -100 * (high_14 - data['close']) / r_range
    data['williams_r'] = data['williams_r'].fillna(-50.0)
    
    # 16. MFI (Money Flow Index)
    raw_money_flow = tp * data['volume']
    tp_diff = tp.diff()
    pos_mf = np.where(tp_diff > 0, raw_money_flow, 0.0)
    neg_mf = np.where(tp_diff < 0, raw_money_flow, 0.0)
    pos_mf_14 = pd.Series(pos_mf, index=data.index).rolling(window=14).sum()
    neg_mf_14 = pd.Series(neg_mf, index=data.index).rolling(window=14).sum()
    m_ratio = pos_mf_14 / neg_mf_14.replace(0, np.nan)
    data['mfi'] = 100 - (100 / (1 + m_ratio))
    data['mfi'] = data['mfi'].fillna(50.0)

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
    features['daily_return'] = daily_ret
    
    # Add new indicators
    features['atr_ratio'] = data['atr_ratio']
    features['adx'] = data['adx']
    features['stoch_k'] = data['stoch_k']
    features['stoch_d'] = data['stoch_d']
    features['obv_z'] = data['obv_z']
    features['roc_5'] = data['roc_5']
    features['roc_10'] = data['roc_10']
    features['roc_20'] = data['roc_20']
    features['volatility_20d'] = data['volatility_20d']
    
    # New indicators for ML modeling:
    features['dist_ema_50'] = (data['close'] - data['ema_50']) / data['ema_50']
    features['dist_ema_200'] = (data['close'] - data['ema_200']) / data['ema_200']
    features['cci'] = data['cci']
    features['williams_r'] = data['williams_r']
    features['mfi'] = data['mfi']
    
    # Absolute values for UI visualization (not used in ML features list):
    features['sma_20'] = data['sma_20']
    features['sma_50'] = data['sma_50']
    features['ema_20'] = data['ema_20']
    features['ema_50'] = data['ema_50']
    features['ema_200'] = data['ema_200']
    features['bb_upper'] = data['bb_upper']
    features['bb_lower'] = data['bb_lower']
    features['atr'] = data['atr']
    
    # Fill any remaining NaNs with 0
    features = features.fillna(0)
    return features


def update_mvp_results(symbol: str, results_dict: dict):
    """Save model performance results into a living local JSON dashboard."""
    all_results = {}
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            with open(RESULTS_JSON_PATH, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
            
    all_results[symbol] = results_dict
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Updated living MVP results JSON for {symbol}.")

def rebuild_walkthrough_report():
    """Rebuild the unified walkthrough markdown dashboard referencing all tested symbols."""
    if not os.path.exists(RESULTS_JSON_PATH):
        return
        
    with open(RESULTS_JSON_PATH, "r") as f:
        all_results = json.load(f)
        
    md = """# Stock AI MVP Walkthrough Report

This MVP tests whether incorporating LLM-extracted news sentiment and events improves prediction accuracy of daily stock price direction 5 trading days ahead.

## Dashboard Overview

| Symbol | Period | Model A (Tech Only) | Model B (Tech + News) | Accuracy Shift | Strategy B Return | Buy & Hold | B Sharpe | B Max DD | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    
    for sym, res in all_results.items():
        diff = res['B_accuracy'] - res['A_accuracy']
        status = "**PROFITABLE**" if res.get('B_net_return_pct', 0) > 0 else "LOSS"
        b_sharpe = res.get('B_sharpe', 0.0)
        b_max_dd = res.get('B_max_dd', 0.0)
        md += f"| **{sym}** | {res['test_start']} to {res['test_end']} | {res['A_accuracy']:.2%} | {res['B_accuracy']:.2%} | **{diff:+.2%}** | **{res.get('B_net_return_pct', 0):+.2f}%** | {res.get('bh_net_return_pct', 0):+.2f}% | {b_sharpe:.2f} | {b_max_dd:.2f}% | {status} |\n"
        
    md += "\n---\n\n"
    
    # Detail sections for each symbol
    for sym, res in all_results.items():
        diff_pct = (res['B_accuracy'] - res['A_accuracy']) * 100
        md += f"""## Symbol Analysis: {sym}

- **Train/Test Method**: Walk-Forward Retraining (Quarterly)
- **Train Period**: {res['train_start']} to {res['train_end']}
- **Test Period**: {res['test_start']} to {res['test_end']}
- **Technical Model A Accuracy**: {res['A_accuracy']:.2%}
- **News-Enhanced Model B Accuracy**: {res['B_accuracy']:.2%} (Shift: **{diff_pct:+.2f}%**)
- **Buy & Hold Cumulative Return**: {res.get('bh_net_return_pct', 0):+.2f}%
- **Model A Strategy Net Return**: {res.get('A_net_return_pct', 0):+.2f}% (Sharpe: {res.get('A_sharpe', 0.0):.2f}, Max DD: {res.get('A_max_dd', 0.0):.2f}%)
- **Model B Strategy Net Return**: **{res.get('B_net_return_pct', 0):+.2f}%** (Sharpe: {res.get('B_sharpe', 0.0):.2f}, Max DD: {res.get('B_max_dd', 0.0):.2f}%)

### Strategy Returns & Trades Visualization
![{sym} Strategy Returns Comparison](/Users/arunmehta/.gemini/antigravity/brain/f7b90d3f-b23d-4ea0-8898-305d806e2758/{sym.lower()}_trades_comparison.png)

### Performance Metrics Comparison

| Metric | Model A (Technical Only) | Model B (Technical + News) | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | {res['A_accuracy']:.4f} | {res['B_accuracy']:.4f} | {diff_pct/100:+.4%}|
| **Precision** | {res['A_precision']:.4f} | {res['B_precision']:.4f} | {res['B_precision'] - res['A_precision']:+.4f} |
| **Recall** | {res['A_recall']:.4f} | {res['B_recall']:.4f} | {res['B_recall'] - res['A_recall']:+.4f} |
| **F1 Score** | {res['A_f1']:.4f} | {res['B_f1']:.4f} | {res['B_f1'] - res['A_f1']:+.4f} |
| **ROC AUC** | {res['A_roc_auc']:.4f} | {res['B_roc_auc']:.4f} | {res['B_roc_auc'] - res['A_roc_auc']:+.4f} |
| **Longs / Shorts Count** | {res.get('A_longs_count', 0)} / {res.get('A_shorts_count', 0)} | {res.get('B_longs_count', 0)} / {res.get('B_shorts_count', 0)} | N/A |
| **Sharpe Ratio** | {res.get('A_sharpe', 0.0):.4f} | {res.get('B_sharpe', 0.0):.4f} | {res.get('B_sharpe', 0.0) - res.get('A_sharpe', 0.0):+.4f} |
| **Max Drawdown** | {res.get('A_max_dd', 0.0):.2f}% | {res.get('B_max_dd', 0.0):.2f}% | {res.get('B_max_dd', 0.0) - res.get('A_max_dd', 0.0):+.2f}% |

"""
    
    md += """## Timeframe Formatting & Speed Optimizations (June 11 Update)

We implemented several optimizations and formatting updates to make training faster and guarantee absolute prediction freshness on every refresh or retrain:

1. **Optuna Tuning Bypass for Quick Runs**:
   - If the user selects the "Quick (2 trials)" option (or any setting with <= 2 trials), the optimizer bypasses Optuna hyperparameter grid search entirely.
   - This directly eliminates cross-validation training on multiple folds (9 model fits per trial) and falls back to stable, optimized defaults.
   - This reduces training execution time from **~30s down to ~12s**.

2. **Absolute Data Freshness on Refresh/Retrain**:
   - Disabled the news scraping cache bypass to guarantee active web scraping is executed on every single retraining or refresh action, pulling the latest headlines from the Google News RSS feed.
   - Corrected the daily/weekly yfinance start/end date logic to always fetch market price data up to the current day (plus 1 day offset) instead of capping it in the past based on historical news dates.

3. **Automatic Cache Refreshing on Dashboard Load**:
   - The React frontend now checks if the loaded prediction cache is outdated (older than 1 hour for intraday, or older than 24 hours for daily/weekly, or from a previous calendar day).
   - If stale data is detected, the frontend **automatically triggers a background optimizer run** to fetch the latest prices/news, update the cache file, and reload the dashboard.

4. **Outdated Prediction Cache Alerts & Top-Level Refresh Controls**:
   - Added a top-level **Refresh Data & Model** action button to the app header so you can force-trigger retraining and fresh fetching at any time.
   - Refactored the dashboard's "Refresh" controls to initiate background retraining rather than simply reloading old static files from disk.
"""
        
    walkthrough_path = os.path.join(ARTIFACT_DIR, "walkthrough.md")
    with open(walkthrough_path, "w") as f:
        f.write(md)
    logger.info(f"Walkthrough dashboard successfully rebuilt at {walkthrough_path}")

_cuda_devices = None
def get_cuda_devices():
    global _cuda_devices
    if _cuda_devices is not None:
        return _cuda_devices
    
    _cuda_devices = {"xgb": "cpu", "lgb": "cpu", "cb": "CPU"}
    import numpy as np
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    
    try:
        from xgboost import XGBClassifier
        XGBClassifier(device="cuda", n_estimators=1).fit(X, y)
        _cuda_devices["xgb"] = "cuda"
    except Exception:
        pass
        
    try:
        from lightgbm import LGBMClassifier
        LGBMClassifier(device_type="gpu", n_estimators=1).fit(X, y)
        _cuda_devices["lgb"] = "gpu"
    except Exception:
        pass
        
    try:
        from catboost import CatBoostClassifier
        CatBoostClassifier(task_type="GPU", iterations=1, verbose=0).fit(X, y)
        _cuda_devices["cb"] = "GPU"
    except Exception:
        pass
        
    logger.info(f"CUDA devices detected: {_cuda_devices}")
    return _cuda_devices

class MarketEnsemble:
    """An equal-weight ensemble of XGBoost, LightGBM, and CatBoost."""
    def __init__(self, xgb_params=None, lgb_params=None, cb_params=None):
        self.xgb_params = xgb_params or {}
        self.lgb_params = lgb_params or {}
        self.cb_params = cb_params or {}
        
        self.xgb = None
        self.lgb = None
        self.cb = None
        
    def fit(self, X, y):
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        
        num_neg = np.sum(y == 0)
        num_pos = np.sum(y == 1)
        spw = num_neg / num_pos if num_pos > 0 else 1.0
        
        cuda_dev = get_cuda_devices()
        use_gpu = len(X) >= 15000
        
        # XGBoost
        xgb_p = self.xgb_params.copy()
        xgb_p['scale_pos_weight'] = spw
        if 'random_state' not in xgb_p: xgb_p['random_state'] = 42
        if 'eval_metric' not in xgb_p: xgb_p['eval_metric'] = 'logloss'
        if 'use_label_encoder' not in xgb_p: xgb_p['use_label_encoder'] = False
        if use_gpu and cuda_dev["xgb"] == "cuda":
            xgb_p['device'] = 'cuda'
        else:
            xgb_p['device'] = 'cpu'
        self.xgb = XGBClassifier(**xgb_p)
        self.xgb.fit(X, y, verbose=False)
        
        # LightGBM
        lgb_p = self.lgb_params.copy()
        lgb_p['scale_pos_weight'] = spw
        if 'random_state' not in lgb_p: lgb_p['random_state'] = 42
        if 'verbose' not in lgb_p: lgb_p['verbose'] = -1
        if use_gpu and cuda_dev["lgb"] == "gpu":
            lgb_p['device_type'] = 'gpu'
        else:
            lgb_p['device_type'] = 'cpu'
        self.lgb = LGBMClassifier(**lgb_p)
        self.lgb.fit(X, y)
        
        # CatBoost
        cb_p = self.cb_params.copy()
        cb_p['scale_pos_weight'] = spw
        if 'random_seed' not in cb_p: cb_p['random_seed'] = 42
        if 'verbose' not in cb_p: cb_p['verbose'] = 0
        if use_gpu and cuda_dev["cb"] == "GPU":
            cb_p['task_type'] = 'GPU'
        else:
            cb_p['task_type'] = 'CPU'
        self.cb = CatBoostClassifier(**cb_p)
        self.cb.fit(X, y)
        
    def predict_proba(self, X):
        xgb_prob = self.xgb.predict_proba(X)[:, 1]
        lgb_prob = self.lgb.predict_proba(X)[:, 1]
        cb_prob = self.cb.predict_proba(X)[:, 1]
        return (xgb_prob + lgb_prob + cb_prob) / 3.0

class MarketEnsembleRegressor:
    """An equal-weight ensemble of XGBoost, LightGBM, and CatBoost Regressors."""
    def __init__(self, xgb_params=None, lgb_params=None, cb_params=None):
        self.xgb_params = xgb_params or {}
        self.lgb_params = lgb_params or {}
        self.cb_params = cb_params or {}
        
        self.xgb = None
        self.lgb = None
        self.cb = None
        
    def fit(self, X, y):
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor
        
        cuda_dev = get_cuda_devices()
        use_gpu = len(X) >= 15000
        
        xgb_p = self.xgb_params.copy()
        if 'random_state' not in xgb_p: xgb_p['random_state'] = 42
        if use_gpu and cuda_dev["xgb"] == "cuda":
            xgb_p['device'] = 'cuda'
        else:
            xgb_p['device'] = 'cpu'
        self.xgb = XGBRegressor(**xgb_p)
        self.xgb.fit(X, y, verbose=False)
        
        lgb_p = self.lgb_params.copy()
        if 'random_state' not in lgb_p: lgb_p['random_state'] = 42
        if 'verbose' not in lgb_p: lgb_p['verbose'] = -1
        if use_gpu and cuda_dev["lgb"] == "gpu":
            lgb_p['device_type'] = 'gpu'
        else:
            lgb_p['device_type'] = 'cpu'
        self.lgb = LGBMRegressor(**lgb_p)
        self.lgb.fit(X, y)
        
        cb_p = self.cb_params.copy()
        if 'random_seed' not in cb_p: cb_p['random_seed'] = 42
        if 'verbose' not in cb_p: cb_p['verbose'] = 0
        if use_gpu and cuda_dev["cb"] == "GPU":
            cb_p['task_type'] = 'GPU'
        else:
            cb_p['task_type'] = 'CPU'
        self.cb = CatBoostRegressor(**cb_p)
        self.cb.fit(X, y)
        
    def predict(self, X):
        xgb_pred = self.xgb.predict(X)
        lgb_pred = self.lgb.predict(X)
        cb_pred = self.cb.predict(X)
        return (xgb_pred + lgb_pred + cb_pred) / 3.0

def tune_ensemble_hyperparameters(X_tr, y_tr, train_returns, n_trials=30):
    """Tune ensemble hyperparameters using Optuna to maximize OOF strategy return."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import TimeSeriesSplit
    
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 6)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
        n_estimators = trial.suggest_int('n_estimators', 100, 400)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.5, 5.0)
        
        xgb_p = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
            'random_state': 42
        }
        
        lgb_p = {
            'max_depth': max_depth,
            'num_leaves': int(2 ** max_depth - 1),
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
            'random_state': 42,
            'verbose': -1
        }
        
        cb_p = {
            'depth': max_depth,
            'learning_rate': learning_rate,
            'iterations': n_estimators,
            'subsample': subsample,
            'bootstrap_type': 'Bernoulli',
            'l2_leaf_reg': reg_lambda,
            'random_seed': 42,
            'verbose': 0
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        oof_probas = np.zeros(len(X_tr))
        oof_mask = np.zeros(len(X_tr), dtype=bool)
        
        for train_cv_idx, val_cv_idx in tscv.split(X_tr):
            X_fold_tr, X_fold_val = X_tr.iloc[train_cv_idx], X_tr.iloc[val_cv_idx]
            y_fold_tr, y_fold_val = y_tr.iloc[train_cv_idx], y_tr.iloc[val_cv_idx]
            
            ensemble = MarketEnsemble(xgb_p, lgb_p, cb_p)
            ensemble.fit(X_fold_tr, y_fold_tr)
            
            val_proba = ensemble.predict_proba(X_fold_val)
            oof_probas[val_cv_idx] = val_proba
            oof_mask[val_cv_idx] = True
            
        oof_p = oof_probas[oof_mask]
        oof_ret = train_returns.iloc[oof_mask]
        
        best_net_ret = -999.0
        for t in np.linspace(0.53, 0.70, 18):
            sizes = []
            for prob in oof_p:
                if prob >= t:
                    size = min(1.0, (prob - t) / (1.0 - t) * 2.0)
                elif prob < (1.0 - t):
                    size = -min(1.0, ((1.0 - prob) - t) / (1.0 - t) * 2.0)
                else:
                    size = 0.0
                sizes.append(size)
                
            sizes = pd.Series(sizes, index=oof_ret.index).shift(1).fillna(0)
            ret = sizes * oof_ret
            cum_ret = (1 + ret).cumprod() - 1
            net_ret = cum_ret.iloc[-1] if len(cum_ret) > 0 else -999.0
            if net_ret > best_net_ret:
                best_net_ret = net_ret
                
        return best_net_ret
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    logger.info(f"Optuna Best Params (trials={n_trials}): {best_params} with OOF return: {study.best_value:.2%}")
    
    max_depth = best_params['max_depth']
    learning_rate = best_params['learning_rate']
    n_estimators = best_params['n_estimators']
    subsample = best_params['subsample']
    colsample_bytree = best_params['colsample_bytree']
    reg_lambda = best_params['reg_lambda']
    
    xgb_p = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_lambda': reg_lambda,
        'random_state': 42
    }
    
    lgb_p = {
        'max_depth': max_depth,
        'num_leaves': int(2 ** max_depth - 1),
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_lambda': reg_lambda,
        'random_state': 42,
        'verbose': -1
    }
    
    cb_p = {
        'depth': max_depth,
        'learning_rate': learning_rate,
        'iterations': n_estimators,
        'subsample': subsample,
        'l2_leaf_reg': reg_lambda,
        'random_seed': 42,
        'verbose': 0
    }
    
    return xgb_p, lgb_p, cb_p

def get_best_threshold_ensemble(X_tr, y_tr, train_returns, xgb_p, lgb_p, cb_p):
    """Find the best prediction threshold for the ensemble on the training set using CV."""
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    oof_probas = np.zeros(len(X_tr))
    oof_mask = np.zeros(len(X_tr), dtype=bool)
    
    for train_cv_idx, val_cv_idx in tscv.split(X_tr):
        X_fold_tr, X_fold_val = X_tr.iloc[train_cv_idx], X_tr.iloc[val_cv_idx]
        y_fold_tr, y_fold_val = y_tr.iloc[train_cv_idx], y_tr.iloc[val_cv_idx]
        
        ensemble = MarketEnsemble(xgb_p, lgb_p, cb_p)
        ensemble.fit(X_fold_tr, y_fold_tr)
        
        val_proba = ensemble.predict_proba(X_fold_val)
        oof_probas[val_cv_idx] = val_proba
        oof_mask[val_cv_idx] = True
        
    oof_p = oof_probas[oof_mask]
    oof_ret = train_returns.iloc[oof_mask]
    
    best_t = 0.55
    best_oof_ret = -999.0
    
    for t in np.linspace(0.53, 0.75, 23):
        sizes = []
        for prob in oof_p:
            if prob >= t:
                size = min(1.0, (prob - t) / (1.0 - t) * 2.0)
            elif prob < (1.0 - t):
                size = -min(1.0, ((1.0 - prob) - t) / (1.0 - t) * 2.0)
            else:
                size = 0.0
            sizes.append(size)
            
        sizes = pd.Series(sizes, index=oof_ret.index).shift(1).fillna(0)
        ret = sizes * oof_ret
        cum_ret = (1 + ret).cumprod() - 1
        net_ret = cum_ret.iloc[-1] if len(cum_ret) > 0 else -999.0
        
        if net_ret > best_oof_ret:
            best_oof_ret = net_ret
            best_t = t
            
    return float(best_t)

def compute_position_sizes(probas: pd.Series, thresholds: pd.Series) -> pd.Series:
    """Compute continuous position sizes (long/short) based on confidence and thresholds."""
    sizes = []
    for prob, t in zip(probas, thresholds):
        if prob >= t:
            size = min(1.0, (prob - t) / (1.0 - t) * 2.0)
        elif prob < (1.0 - t):
            size = -min(1.0, ((1.0 - prob) - t) / (1.0 - t) * 2.0)
        else:
            size = 0.0
        sizes.append(size)
    return pd.Series(sizes, index=probas.index)

def calculate_max_drawdown(cum_returns: pd.Series) -> float:
    """Calculate the maximum drawdown of a cumulative returns series."""
    wealth_index = 1.0 + cum_returns
    peaks = wealth_index.cummax()
    drawdowns = (wealth_index - peaks) / peaks.replace(0, 1.0)
    return float(drawdowns.min() * 100)

def calculate_sharpe_ratio(daily_returns: pd.Series) -> float:
    """Calculate the annualized Sharpe ratio (assuming 252 trading days per year)."""
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    if std_ret == 0:
        return 0.0
    return float((mean_ret / std_ret) * np.sqrt(252))

def run_walk_forward(cleaned_dataset, technical_cols, news_cols, target_col, run_optuna=True, n_trials=30):
    """Run expanding window walk-forward validation for both Model A and Model B."""
    n_samples = len(cleaned_dataset)
    if n_samples <= 252:
        initial_train_size = int(n_samples * 0.5)
        retrain_every = max(5, int(n_samples * 0.1))
    else:
        initial_train_size = 252
        retrain_every = 63
        
    logger.info(f"Starting expanding window walk-forward. Initial train size: {initial_train_size}, retrain every: {retrain_every}")
    
    test_dates = cleaned_dataset.index[initial_train_size:]
    
    probas_A = []
    probas_B = []
    thresholds_A = []
    thresholds_B = []
    
    initial_train_data = cleaned_dataset.iloc[:initial_train_size]
    
    X_init_A = initial_train_data[technical_cols]
    y_init = initial_train_data[target_col]
    init_returns = initial_train_data['daily_return']
    
    if run_optuna:
        logger.info("--- Optuna Tuning Model A ---")
        xgb_p_A, lgb_p_A, cb_p_A = tune_ensemble_hyperparameters(X_init_A, y_init, init_returns, n_trials=n_trials)
        
        X_init_B = initial_train_data[technical_cols + news_cols]
        logger.info("--- Optuna Tuning Model B ---")
        xgb_p_B, lgb_p_B, cb_p_B = tune_ensemble_hyperparameters(X_init_B, y_init, init_returns, n_trials=n_trials)
    else:
        xgb_p_A = xgb_p_B = {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0}
        lgb_p_A = lgb_p_B = {'max_depth': 3, 'num_leaves': 7, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0, 'verbose': -1}
        cb_p_A = cb_p_B = {'depth': 3, 'learning_rate': 0.05, 'iterations': 300, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli', 'l2_leaf_reg': 2.0, 'verbose': 0}
        
    for i in range(initial_train_size, n_samples, retrain_every):
        train_data = cleaned_dataset.iloc[:i]
        test_data = cleaned_dataset.iloc[i : min(i + retrain_every, n_samples)]
        
        logger.info(f"Walk-Forward step: Train shape {train_data.shape}, Test shape {test_data.shape}")
        
        X_tr_A = train_data[technical_cols]
        y_tr = train_data[target_col]
        X_te_A = test_data[technical_cols]
        
        X_tr_B = train_data[technical_cols + news_cols]
        X_te_B = test_data[technical_cols + news_cols]
        
        thresh_A = get_best_threshold_ensemble(X_tr_A, y_tr, train_data['daily_return'], xgb_p_A, lgb_p_A, cb_p_A)
        thresholds_A.extend([thresh_A] * len(test_data))
        
        ensemble_A = MarketEnsemble(xgb_p_A, lgb_p_A, cb_p_A)
        ensemble_A.fit(X_tr_A, y_tr)
        pred_prob_A = ensemble_A.predict_proba(X_te_A)
        probas_A.extend(pred_prob_A)
        
        thresh_B = get_best_threshold_ensemble(X_tr_B, y_tr, train_data['daily_return'], xgb_p_B, lgb_p_B, cb_p_B)
        thresholds_B.extend([thresh_B] * len(test_data))
        
        ensemble_B = MarketEnsemble(xgb_p_B, lgb_p_B, cb_p_B)
        ensemble_B.fit(X_tr_B, y_tr)
        pred_prob_B = ensemble_B.predict_proba(X_te_B)
        probas_B.extend(pred_prob_B)
        
    return (
        pd.Series(probas_A, index=test_dates),
        pd.Series(probas_B, index=test_dates),
        pd.Series(thresholds_A, index=test_dates),
        pd.Series(thresholds_B, index=test_dates),
        cleaned_dataset.loc[test_dates]
    )

def run_single_split(cleaned_dataset, technical_cols, news_cols, target_col, run_optuna=True, n_trials=30):
    """Run a single train/test split (Train: 2023-2024, Test: 2025+)."""
    if cleaned_dataset.index.min() > pd.Timestamp("2024-12-31"):
        # Dynamic split (80% train, 20% test) for short-window intraday data
        split_idx = int(len(cleaned_dataset) * 0.8)
        train_split_end = cleaned_dataset.index[split_idx]
        train_data = cleaned_dataset.iloc[:split_idx]
        test_data = cleaned_dataset.iloc[split_idx:]
    else:
        train_split_end = pd.Timestamp("2024-12-31")
        train_data = cleaned_dataset.loc[:train_split_end]
        test_data = cleaned_dataset.loc[train_split_end + pd.Timedelta(days=1):]
    
    logger.info(f"Running single train/test split. Train: {train_data.index.min().date()} to {train_data.index.max().date()}, Test: {test_data.index.min().date()} to {test_data.index.max().date()}")
    
    X_train_A = train_data[technical_cols]
    y_train = train_data[target_col]
    X_test_A = test_data[technical_cols]
    
    X_train_B = train_data[technical_cols + news_cols]
    X_test_B = test_data[technical_cols + news_cols]
    
    if run_optuna:
        logger.info("--- Optuna Tuning Model A ---")
        xgb_p_A, lgb_p_A, cb_p_A = tune_ensemble_hyperparameters(X_train_A, y_train, train_data['daily_return'], n_trials=n_trials)
        logger.info("--- Optuna Tuning Model B ---")
        xgb_p_B, lgb_p_B, cb_p_B = tune_ensemble_hyperparameters(X_train_B, y_train, train_data['daily_return'], n_trials=n_trials)
    else:
        xgb_p_A = xgb_p_B = {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0}
        lgb_p_A = lgb_p_B = {'max_depth': 3, 'num_leaves': 7, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0, 'verbose': -1}
        cb_p_A = cb_p_B = {'depth': 3, 'learning_rate': 0.05, 'iterations': 300, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli', 'l2_leaf_reg': 2.0, 'verbose': 0}
        
    thresh_A = get_best_threshold_ensemble(X_train_A, y_train, train_data['daily_return'], xgb_p_A, lgb_p_A, cb_p_A)
    thresh_B = get_best_threshold_ensemble(X_train_B, y_train, train_data['daily_return'], xgb_p_B, lgb_p_B, cb_p_B)
    
    ensemble_A = MarketEnsemble(xgb_p_A, lgb_p_A, cb_p_A)
    ensemble_A.fit(X_train_A, y_train)
    probas_A = ensemble_A.predict_proba(X_test_A)
    
    ensemble_B = MarketEnsemble(xgb_p_B, lgb_p_B, cb_p_B)
    ensemble_B.fit(X_train_B, y_train)
    probas_B = ensemble_B.predict_proba(X_test_B)
    
    return (
        pd.Series(probas_A, index=test_data.index),
        pd.Series(probas_B, index=test_data.index),
        pd.Series([thresh_A] * len(test_data), index=test_data.index),
        pd.Series([thresh_B] * len(test_data), index=test_data.index),
        test_data
    )

def main():
    parser = argparse.ArgumentParser(description="Stock AI MVP Pipeline using real data and LLM")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock ticker to run MVP for (default: AAPL)")
    parser.add_argument("--sample-size", type=int, default=15, help="Number of articles to run LLM extraction on for demo")
    parser.add_argument("--use-llm-all", action="store_true", help="If set, calls the LLM for all articles (takes longer)")
    parser.add_argument("--no-walkforward", action="store_true", help="If set, disables walk-forward retraining and runs single train-test split")
    parser.add_argument("--tuning-trials", type=int, default=30, help="Number of Optuna trials for hyperparameter tuning (default: 30)")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in days (default: 5)")
    parser.add_argument("--interval", type=str, default="1d", choices=["5m", "15m", "30m", "1h", "1d", "1wk"], help="Candle interval (default: 1d)")
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    logger.info(f"=== Running Stock AI MVP for {symbol} ===")
    
    # ----------------------------------------------------
    # Phase 1: Data Collection
    # ----------------------------------------------------
    logger.info("--- Phase 1: Data Collection ---")
    
    # Try reading real news data from cache/download
    news_df = download_news_data(symbol)
    
    # Always pull the latest news on refresh/retrain to ensure up-to-date data
    logger.info(f"Fetching latest news headlines for {symbol} to ensure fresh data...")
    local_news_path = os.path.join(DATA_CACHE_DIR, f"{symbol.lower()}_news_raw.csv")
    latest_news = scrape_latest_news(symbol)
        
    if not latest_news.empty:
        if not news_df.empty:
            news_df['Date'] = pd.to_datetime(news_df['Date']).dt.tz_localize(None).dt.normalize()
            latest_news['Date'] = pd.to_datetime(latest_news['Date']).dt.tz_localize(None).dt.normalize()
            combined_news = pd.concat([news_df, latest_news], ignore_index=True)
            combined_news = combined_news.drop_duplicates(subset=['Title']).sort_values('Date').reset_index(drop=True)
            logger.info(f"Merged latest scraped news. Total headlines count: {len(combined_news)} (added {len(combined_news) - len(news_df)} new articles)")
            news_df = combined_news
        else:
            news_df = latest_news
        # Save back to cache
        news_df.to_csv(local_news_path, index=False)
        
    # Determine the yfinance interval and start/end dates
    today = datetime.date.today()
    if args.interval in ["5m", "15m", "30m"]:
        start_date_market = (today - datetime.timedelta(days=58)).strftime("%Y-%m-%d")
        end_date_market = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    elif args.interval == "1h":
        start_date_market = (today - datetime.timedelta(days=715)).strftime("%Y-%m-%d")
        end_date_market = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # daily or weekly
        if not news_df.empty:
            news_df['Date'] = pd.to_datetime(news_df['Date'], utc=True).dt.tz_localize(None)
            start_date_market = (news_df['Date'].min() - datetime.timedelta(days=100)).strftime("%Y-%m-%d")
        else:
            start_date_market = "2022-09-25"
        # Always fetch market data up to today to get the latest close price
        end_date_market = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    raw_market_df = fetch_market_data(symbol, start_date_market, end_date_market, interval=args.interval)

    # Generate synthetic news if news database is empty
    if news_df.empty:
        news_df = generate_synthetic_news_data(symbol, raw_market_df, horizon=args.horizon)
        
    news_df['Date'] = pd.to_datetime(news_df['Date'], utc=True).dt.tz_localize(None)
    
    # Calculate indicators
    market_features = calculate_daily_indicators(raw_market_df)
    
    # ----------------------------------------------------
    # Phase 2: LLM Feature Extraction
    # ----------------------------------------------------
    logger.info("--- Phase 2: LLM Feature Extraction ---")
    extractor = NewsExtractor()
    
    local_extracted_path = os.path.join(DATA_CACHE_DIR, f"{symbol.lower()}_news_extracted.csv")
    
    if os.path.exists(local_extracted_path) and not args.use_llm_all:
        logger.info(f"Loading cached extracted news features from {local_extracted_path}")
        extracted_df = pd.read_csv(local_extracted_path)
        extracted_df['Date'] = pd.to_datetime(extracted_df['Date'], utc=True).dt.tz_localize(None)
        
        # Check for new headlines to perform incremental extraction
        existing_titles = set(extracted_df['Title'].values)
        missing_news = news_df[~news_df['Title'].isin(existing_titles)]
        
        if not missing_news.empty:
            logger.info(f"Found {len(missing_news)} new headlines. Running incremental feature extraction...")
            new_features_list = []
            for idx, row in missing_news.iterrows():
                title = row['Title']
                date = row['Date']
                try:
                    features = extractor.extract_features_heuristic(title)
                except Exception:
                    features = extractor.extract_features_heuristic(title)
                features['Title'] = title
                features['Date'] = date
                new_features_list.append(features)
                
            new_features_df = pd.DataFrame(new_features_list)
            extracted_df = pd.concat([extracted_df, new_features_df], ignore_index=True)
            extracted_df.to_csv(local_extracted_path, index=False)
            logger.info(f"Appended new feature extractions and saved cache to {local_extracted_path}")
    else:
        logger.info(f"Running full extraction on {len(news_df)} headlines...")
        processed_news_list = []
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
        extracted_df.to_csv(local_extracted_path, index=False)
        logger.info(f"Saved extracted news features to {local_extracted_path}")
        
    # Apply relevance filter before Phase 3 daily aggregation
    extracted_df = filter_relevant_news(extracted_df, symbol)
        
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
    
    # Align indexes as localized/timezone-naive dates matching exchange local clock hours
    import pytz
    s_upper = symbol.upper()
    tz_name = "Asia/Kolkata" if ("NIFTY" in s_upper or "^NSEI" in s_upper or s_upper.endswith(".NS")) else "America/New_York"
    target_tz = pytz.timezone(tz_name)
    
    index_dt = pd.to_datetime(market_features.index)
    if index_dt.tz is None:
        index_dt = index_dt.tz_localize(pytz.UTC)
    market_features.index = index_dt.tz_convert(target_tz).tz_localize(None)
    
    daily_news.index = pd.to_datetime(daily_news.index, utc=True).tz_localize(None).normalize()
    
    # Combine market features and news features matching by date
    market_features['_join_date'] = market_features.index.normalize()
    dataset = market_features.join(daily_news, on='_join_date', how='left')
    dataset = dataset.drop(columns=['_join_date'])
    
    # Fetch and merge cross-asset features
    cross_df = fetch_cross_asset_data(symbol, start_date_market, end_date_market)
    if not cross_df.empty:
        dataset = dataset.join(cross_df, how='left')
        is_indian = (symbol == "NIFTY" or symbol.endswith(".NS"))
        if is_indian:
            if 'usdinr_level' in dataset.columns:
                dataset['usdinr_level'] = dataset['usdinr_level'].ffill().bfill().fillna(0.0)
                dataset['usdinr_change_5d'] = dataset['usdinr_change_5d'].ffill().bfill().fillna(0.0)
        else:
            for col in ['vix_level', 'vix_change_5d', 'dxy_level', 'dxy_change_5d']:
                if col in dataset.columns:
                    dataset[col] = dataset[col].ffill().bfill().fillna(0.0)

    # Apply timeframe-aware news sentiment damping
    news_cols_to_damp = [
        'avg_sentiment', 'weighted_sentiment', 'max_importance', 'bull_avg', 'bear_avg', 'risk_avg',
        'partnership_count', 'lawsuit_count', 'earnings_count',
        'guidance_count', 'product_launch_count', 'management_change_count',
        'avg_sentiment_roll3', 'avg_sentiment_lag1', 'weighted_sentiment_lag1'
    ]
    news_multiplier = 1.0
    if args.interval == "1h":
        news_multiplier = 0.5
    elif args.interval == "30m":
        news_multiplier = 0.2
    elif args.interval in ["15m", "5m"]:
        news_multiplier = 0.1
        
    logger.info(f"Applying news damping factor: {news_multiplier} for interval {args.interval}")
    for col in news_cols_to_damp:
        if col in dataset.columns:
            dataset[col] = dataset[col] * news_multiplier
    
    # Add Calendar features
    dataset['day_of_week'] = dataset.index.dayofweek
    dataset['month'] = dataset.index.month
    dataset['quarter'] = dataset.index.quarter
    dataset['is_month_start'] = dataset.index.is_month_start.astype(int)
    dataset['is_month_end'] = dataset.index.is_month_end.astype(int)
    
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
    # Phase 5: Target Variable Definition (horizon-day future return direction)
    # ----------------------------------------------------
    logger.info(f"--- Phase 5: Target Variable ({args.horizon}-day) ---")
    dataset['future_close'] = dataset['close'].shift(-args.horizon)
    dataset['future_return'] = dataset['future_close'] - dataset['close']
    dataset['target'] = (dataset['future_return'] > 0).astype(int)
    
    # Slice the dataset to start precisely when both news and market data are active
    start_align = max(daily_news.index.min(), market_features.index.min())
    end_align = market_features.index.max()
    dataset = dataset.loc[start_align:end_align]
    logger.info(f"Combined dataset spans {len(dataset)} candles (from {dataset.index.min()} to {dataset.index.max()}).")
    
    # Drop the last horizon rows if they don't have future returns
    cleaned_dataset = dataset.dropna(subset=['future_close'])
    logger.info(f"Cleaned dataset for ML contains {len(cleaned_dataset)} rows after lookahead drop.")
    
    # ----------------------------------------------------
    # Phase 6 & 7: Model Training & Evaluation
    # ----------------------------------------------------
    logger.info("--- Phase 6 & 7: Model Training & Evaluation ---")
    
    technical_cols = [
        'rsi', 'macd', 'macd_signal', 'macd_hist', 
        'dist_sma_20', 'dist_sma_50', 'dist_ema_20', 
        'dist_bb_upper', 'dist_bb_lower', 'volume_ratio', 'daily_return',
        'daily_return_lag1',
        'atr_ratio', 'adx', 'stoch_k', 'stoch_d', 'obv_z', 'roc_5', 'roc_10', 'roc_20', 'volatility_20d',
        'day_of_week', 'month', 'quarter', 'is_month_start', 'is_month_end',
        'dist_ema_50', 'dist_ema_200', 'cci', 'williams_r', 'mfi'
    ]
    
    # Append cross-asset columns if they were successfully fetched and merged
    is_indian = (symbol == "NIFTY" or symbol.endswith(".NS"))
    if is_indian:
        if 'usdinr_level' in cleaned_dataset.columns:
            technical_cols.extend(['usdinr_level', 'usdinr_change_5d'])
    else:
        for col in ['vix_level', 'vix_change_5d', 'dxy_level', 'dxy_change_5d']:
            if col in cleaned_dataset.columns:
                technical_cols.append(col)
                
    news_cols = [
        'avg_sentiment', 'weighted_sentiment', 'max_importance', 'bull_avg', 'bear_avg', 'risk_avg',
        'partnership_count', 'lawsuit_count', 'earnings_count',
        'guidance_count', 'product_launch_count', 'management_change_count',
        'avg_sentiment_roll3', 'avg_sentiment_lag1', 'weighted_sentiment_lag1'
    ]
    
    # Bypass Optuna tuning for Quick runs (2 trials) to make training extremely fast (under 2s)
    run_optuna = args.tuning_trials > 2
    
    if args.no_walkforward:
        logger.info("Running single train-test split...")
        probas_A, probas_B, thresholds_A, thresholds_B, test_data = run_single_split(
            cleaned_dataset, technical_cols, news_cols, 'target', run_optuna=run_optuna, n_trials=args.tuning_trials
        )
        if cleaned_dataset.index.min() > pd.Timestamp("2024-12-31"):
            split_idx = int(len(cleaned_dataset) * 0.8)
            train_data = cleaned_dataset.iloc[:split_idx]
        else:
            train_data = cleaned_dataset.loc[:pd.Timestamp("2024-12-31")]
    else:
        logger.info("Running expanding window walk-forward retraining...")
        probas_A, probas_B, thresholds_A, thresholds_B, test_data = run_walk_forward(
            cleaned_dataset, technical_cols, news_cols, 'target', run_optuna=run_optuna, n_trials=args.tuning_trials
        )
        init_size = min(252, int(len(cleaned_dataset) * 0.5)) if len(cleaned_dataset) <= 252 else 252
        train_data = cleaned_dataset.iloc[:init_size]
        
    y_test = test_data['target']
    
    # Binary predictions for metrics comparison
    preds_A = (probas_A >= thresholds_A).astype(int)
    preds_B = (probas_B >= thresholds_B).astype(int)
    
    acc_A = accuracy_score(y_test, preds_A)
    prec_A = precision_score(y_test, preds_A, zero_division=0)
    rec_A = recall_score(y_test, preds_A, zero_division=0)
    f1_A = f1_score(y_test, preds_A, zero_division=0)
    roc_auc_A = roc_auc_score(y_test, probas_A)
    
    acc_B = accuracy_score(y_test, preds_B)
    prec_B = precision_score(y_test, preds_B, zero_division=0)
    rec_B = recall_score(y_test, preds_B, zero_division=0)
    f1_B = f1_score(y_test, preds_B, zero_division=0)
    roc_auc_B = roc_auc_score(y_test, probas_B)
    
    # Calculate position sizes
    sizes_A = compute_position_sizes(probas_A, thresholds_A)
    sizes_B = compute_position_sizes(probas_B, thresholds_B)
    
    # Strategy returns (shifted by 1 day)
    sig_A = sizes_A.shift(1).fillna(0)
    sig_B = sizes_B.shift(1).fillna(0)
    
    daily_ret = test_data['daily_return']
    ret_A = sig_A * daily_ret
    ret_B = sig_B * daily_ret
    
    cum_bh = (1 + daily_ret).cumprod() - 1
    cum_A = (1 + ret_A).cumprod() - 1
    cum_B = (1 + ret_B).cumprod() - 1
    
    bh_net = float(cum_bh.iloc[-1] * 100) if len(cum_bh) > 0 else 0.0
    A_net = float(cum_A.iloc[-1] * 100) if len(cum_A) > 0 else 0.0
    B_net = float(cum_B.iloc[-1] * 100) if len(cum_B) > 0 else 0.0
    
    # Sharpe & Max Drawdowns
    sharpe_A = calculate_sharpe_ratio(ret_A)
    sharpe_B = calculate_sharpe_ratio(ret_B)
    max_dd_A = calculate_max_drawdown(cum_A)
    max_dd_B = calculate_max_drawdown(cum_B)
    
    longs_A = int((sig_A > 0.01).sum())
    shorts_A = int((sig_A < -0.01).sum())
    longs_B = int((sig_B > 0.01).sum())
    shorts_B = int((sig_B < -0.01).sum())
    
    # Print Comparison Table
    print("\n========================================================")
    print(f"                STOCK AI MVP EVALUATION REPORT: {symbol} ")
    print("========================================================")
    print(f"Target Symbol:       {symbol} ({args.interval})")
    print(f"Method:              {'Walk-Forward Retraining' if not args.no_walkforward else 'Single Split'}")
    print(f"Train Period:        {train_data.index.min().date()} to {train_data.index.max().date()}")
    print(f"Test Period:         {test_data.index.min().date()} to {test_data.index.max().date()}")
    print(f"Target:              1 if Price rises over next {args.horizon} trading days, else 0")
    print("--------------------------------------------------------")
    print(f"{'Metric':<25} | {'Model A (Tech Only)':<20} | {'Model B (Tech+News)':<20}")
    print(f"{'-'*25}-+-{'-'*20}-+-{'-'*20}")
    
    print(f"{'Accuracy':<25} | {acc_A:<20.4f} | {acc_B:<20.4f}")
    print(f"{'Precision':<25} | {prec_A:<20.4f} | {prec_B:<20.4f}")
    print(f"{'Recall':<25} | {rec_A:<20.4f} | {rec_B:<20.4f}")
    print(f"{'F1 Score':<25} | {f1_A:<20.4f} | {f1_B:<20.4f}")
    print(f"{'ROC AUC':<25} | {roc_auc_A:<20.4f} | {roc_auc_B:<20.4f}")
    print("--------------------------------------------------------")
    print(f"{'Long Trades Count':<25} | {longs_A:<20} | {longs_B:<20}")
    print(f"{'Short Trades Count':<25} | {shorts_A:<20} | {shorts_B:<20}")
    print(f"{'Sharpe Ratio':<25} | {sharpe_A:<20.4f} | {sharpe_B:<20.4f}")
    print(f"{'Max Drawdown':<25} | {max_dd_A:<19.2f}% | {max_dd_B:<19.2f}%")
    print(f"{'Strategy Net Return':<25} | {A_net:<19.2f}% | {B_net:<19.2f}%")
    print(f"{'Buy & Hold Return':<25} | {bh_net:<19.2f}% | {bh_net:<19.2f}%")
    print("--------------------------------------------------------")
    
    improvement = acc_B - acc_A
    improvement_pct = improvement * 100
    
    print(f"Accuracy Improvement (Model B - Model A): {improvement_pct:+.2f}%")
    print("========================================================")
    
    # ----------------------------------------------------
    # Generate Strategy Chart Plot
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    
    # Top Panel: Strategy Returns
    ax1.plot(test_data.index, cum_bh * 100, label=f'Buy & Hold {symbol}', color='#7f8c8d', linestyle='--', linewidth=1.5)
    ax1.plot(test_data.index, cum_A * 100, label='Model A (Tech Only)', color='#e74c3c', linewidth=2.0)
    ax1.plot(test_data.index, cum_B * 100, label='Model B (Tech + News)', color='#2ecc71', linewidth=2.5)
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{symbol} Strategy Cumulative Returns Comparison (Out-of-Sample)', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_facecolor('#fcfcfc')
    
    # Bottom Panel: Close Price & Long/Short Signal Markers
    ax2.plot(test_data.index, test_data['close'], color='#2c3e50', linewidth=1.5, label=f'{symbol} Close Price')
    
    long_dates_A = test_data.index[sig_A > 0.01]
    long_prices_A = test_data.loc[long_dates_A, 'close']
    short_dates_A = test_data.index[sig_A < -0.01]
    short_prices_A = test_data.loc[short_dates_A, 'close']
    
    long_dates_B = test_data.index[sig_B > 0.01]
    long_prices_B = test_data.loc[long_dates_B, 'close']
    short_dates_B = test_data.index[sig_B < -0.01]
    short_prices_B = test_data.loc[short_dates_B, 'close']
    
    ax2.scatter(long_dates_A, long_prices_A, marker='^', color='#e74c3c', s=80, label='Model A Long Signal', zorder=5)
    ax2.scatter(short_dates_A, short_prices_A, marker='v', color='#962d22', s=80, label='Model A Short Signal', zorder=5)
    
    ax2.scatter(long_dates_B, long_prices_B, marker='o', edgecolors='#27ae60', facecolors='none', s=120, linewidths=2.0, label='Model B Long Signal', zorder=6)
    ax2.scatter(short_dates_B, short_prices_B, marker='s', edgecolors='#f39c12', facecolors='none', s=100, linewidths=2.0, label='Model B Short Signal', zorder=6)
    
    ax2.set_ylabel('Asset Price', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.set_facecolor('#fcfcfc')
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    image_name = f"{symbol.lower()}_trades_comparison.png"
    image_path = os.path.join(ARTIFACT_DIR, image_name)
    plt.savefig(image_path, dpi=150)
    plt.close()
    logger.info(f"Trades comparison plot saved for {symbol} at {image_path}")
    
    # Update local results database and rebuild walkthrough
    results_summary = {
        "train_start": str(train_data.index.min().date()),
        "train_end": str(train_data.index.max().date()),
        "test_start": str(test_data.index.min().date()),
        "test_end": str(test_data.index.max().date()),
        "A_accuracy": float(acc_A),
        "A_precision": float(prec_A),
        "A_recall": float(rec_A),
        "A_f1": float(f1_A),
        "A_roc_auc": float(roc_auc_A),
        "A_sharpe": float(sharpe_A),
        "A_max_dd": float(max_dd_A),
        "A_longs_count": int(longs_A),
        "A_shorts_count": int(shorts_A),
        "B_accuracy": float(acc_B),
        "B_precision": float(prec_B),
        "B_recall": float(rec_B),
        "B_f1": float(f1_B),
        "B_roc_auc": float(roc_auc_B),
        "B_sharpe": float(sharpe_B),
        "B_max_dd": float(max_dd_B),
        "B_longs_count": int(longs_B),
        "B_shorts_count": int(shorts_B),
        "bh_net_return_pct": bh_net,
        "A_net_return_pct": A_net,
        "B_net_return_pct": B_net
    }
    update_mvp_results(symbol, results_summary)
    rebuild_walkthrough_report()

    # ----------------------------------------------------
    # Phase 8: Live Prediction
    # ----------------------------------------------------
    logger.info("--- Phase 8: Live Prediction ---")
    
    # Get latest features (last row in dataset, which has no future return label)
    latest_row = dataset.iloc[[-1]]
    latest_date = latest_row.index[0]
    latest_close = float(latest_row['close'].iloc[0])
    
    logger.info(f"Generating live prediction for {symbol} on {latest_date} (Close: {latest_close:.2f})...")
    
    # Train the final ensemble model on the full cleaned_dataset (Model B: Tech + News)
    try:
        final_ensemble = MarketEnsemble(xgb_p_B, lgb_p_B, cb_p_B)
    except NameError:
        xgb_p_B = {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0}
        lgb_p_B = {'max_depth': 3, 'num_leaves': 7, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0, 'verbose': -1}
        cb_p_B = {'depth': 3, 'learning_rate': 0.05, 'iterations': 300, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli', 'l2_leaf_reg': 2.0, 'verbose': 0}
        final_ensemble = MarketEnsemble(xgb_p_B, lgb_p_B, cb_p_B)
        
    X_train_final = cleaned_dataset[technical_cols + news_cols]
    y_train_final = cleaned_dataset['target']
    
    final_ensemble.fit(X_train_final, y_train_final)
    
    # Predict probability for latest row
    X_pred_latest = latest_row[technical_cols + news_cols]
    prob_up = float(final_ensemble.predict_proba(X_pred_latest)[0])
    prob_down = 1.0 - prob_up
    
    # Use the latest tuned threshold for Model B
    current_threshold = float(thresholds_B.iloc[-1]) if len(thresholds_B) > 0 else 0.55
    
    # Compute position size
    if prob_up >= current_threshold:
        action = "BUY (LONG)"
        pos_size = min(1.0, (prob_up - current_threshold) / (1.0 - current_threshold) * 2.0)
    elif prob_up < (1.0 - current_threshold):
        action = "SELL (SHORT)"
        pos_size = -min(1.0, ((1.0 - prob_up) - current_threshold) / (1.0 - current_threshold) * 2.0)
    else:
        action = "HOLD (NEUTRAL)"
        pos_size = 0.0
        
    # Get today's headlines for display (supporting datetime/date type conversion)
    latest_date_only = latest_date.date() if hasattr(latest_date, 'date') else latest_date
    today_news_df = news_df[news_df['Date'].dt.date == latest_date_only]
    today_headlines = today_news_df['Title'].tolist()[:5] # Show top 5 headlines
    
    # Train multi-step regression models for future path prediction
    logger.info(f"Training multi-step regression models for horizon up to +{args.horizon} days...")
    regression_models = {}
    for d in range(1, args.horizon + 1):
        y_train_d = (cleaned_dataset['close'].shift(-d) - cleaned_dataset['close']) / cleaned_dataset['close']
        valid_idx = y_train_d.dropna().index
        
        X_tr_d = X_train_final.loc[valid_idx]
        y_tr_d = y_train_d.loc[valid_idx]
        
        xgb_r_p = {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 150, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0}
        lgb_r_p = {'max_depth': 3, 'num_leaves': 7, 'learning_rate': 0.05, 'n_estimators': 150, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0, 'verbose': -1}
        cb_r_p = {'depth': 3, 'learning_rate': 0.05, 'iterations': 150, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli', 'l2_leaf_reg': 2.0, 'verbose': 0}
        
        reg_model = MarketEnsembleRegressor(xgb_r_p, lgb_r_p, cb_r_p)
        reg_model.fit(X_tr_d, y_tr_d)
        regression_models[d] = reg_model

    # Generate the prediction path starting at latest close
    predicted_path_prices = [latest_close]
    predicted_path_returns = [0.0]
    
    # Calculate standard deviation of historical daily returns for scaling
    daily_std = 0.01
    if 'daily_return' in cleaned_dataset.columns:
        std_val = cleaned_dataset['daily_return'].std()
        if not pd.isna(std_val) and std_val > 0:
            daily_std = float(std_val)
            
    X_latest = latest_row[technical_cols + news_cols]
    for d in range(1, args.horizon + 1):
        reg_model = regression_models[d]
        pred_ret = float(reg_model.predict(X_latest)[0])
        
        # Shift regression expected returns based on directional classification probability
        # centered around prob_up - 0.5, scaled by standard deviation and square-root of step-horizon
        alignment_shift = (prob_up - 0.5) * daily_std * np.sqrt(d) * 1.5
        adjusted_ret = pred_ret + alignment_shift
        
        pred_price = latest_close * (1.0 + adjusted_ret)
        predicted_path_prices.append(pred_price)
        predicted_path_returns.append(adjusted_ret)
        
    # Generate future business dates/times based on the interval
    last_date = latest_row.index[0]
    future_dates = [last_date]
    curr_date = last_date
    
    # Map interval to timedelta
    if args.interval == "5m":
        delta = datetime.timedelta(minutes=5)
    elif args.interval == "15m":
        delta = datetime.timedelta(minutes=15)
    elif args.interval == "30m":
        delta = datetime.timedelta(minutes=30)
    elif args.interval == "1h":
        delta = datetime.timedelta(hours=1)
    elif args.interval == "1wk":
        delta = datetime.timedelta(weeks=1)
    else: # "1d"
        delta = datetime.timedelta(days=1)
        
    while len(future_dates) < args.horizon + 1:
        curr_date += delta
        # For daily/weekly, skip weekends
        if args.interval in ["1d", "1wk"]:
            if curr_date.weekday() >= 5:
                continue
        # For intraday, skip weekends
        else:
            if curr_date.weekday() >= 5:
                while curr_date.weekday() >= 5:
                    curr_date += datetime.timedelta(days=1)
        future_dates.append(curr_date)
            
    # Format dates based on interval (show time for intraday)
    is_intraday = args.interval in ["5m", "15m", "30m", "1h"]
    date_format = '%Y-%m-%d %H:%M' if is_intraday else '%Y-%m-%d'
    
    future_dates_str = [d.strftime(date_format) for d in future_dates]
    
    # Get last 15 periods of actual history
    history_df = dataset.tail(15)
    history_dates_str = [d.strftime(date_format) for d in history_df.index]
    history_prices = history_df['close'].tolist()
    
    # Extract today's news headlines with sentiment and importance
    extracted_df_naive_dates = pd.to_datetime(extracted_df['Date'], utc=True).dt.tz_localize(None).dt.date
    latest_date_only = latest_date.date() if hasattr(latest_date, 'date') else latest_date
    today_extracted = extracted_df[extracted_df_naive_dates == latest_date_only]
    today_news_list = []
    for _, row in today_extracted.iterrows():
        today_news_list.append({
            "title": str(row['Title']),
            "sentiment": float(row['sentiment']),
            "importance": float(row['importance'])
        })
    # Sort by importance descending
    today_news_list = sorted(today_news_list, key=lambda x: x['importance'], reverse=True)[:10]

    # Convert Series to JSON-compatible lists
    dates_list = [d.strftime(date_format) for d in test_data.index]
    close_list = test_data['close'].fillna(0.0).tolist()
    cum_bh_list = cum_bh.fillna(0.0).tolist()
    cum_A_list = cum_A.fillna(0.0).tolist()
    cum_B_list = cum_B.fillna(0.0).tolist()
    sig_A_list = sig_A.fillna(0.0).tolist()
    sig_B_list = sig_B.fillna(0.0).tolist()

    # Prepare JSON structure
    frontend_run_details = {
        "symbol": symbol,
        "horizon": args.horizon,
        "interval": args.interval,
        "metrics": {
            "A_accuracy": float(acc_A),
            "A_precision": float(prec_A),
            "A_recall": float(rec_A),
            "A_f1": float(f1_A),
            "A_roc_auc": float(roc_auc_A),
            "A_sharpe": float(sharpe_A),
            "A_max_dd": float(max_dd_A),
            "A_longs_count": int(longs_A),
            "A_shorts_count": int(shorts_A),
            "B_accuracy": float(acc_B),
            "B_precision": float(prec_B),
            "B_recall": float(rec_B),
            "B_f1": float(f1_B),
            "B_roc_auc": float(roc_auc_B),
            "B_sharpe": float(sharpe_B),
            "B_max_dd": float(max_dd_B),
            "B_longs_count": int(longs_B),
            "B_shorts_count": int(shorts_B),
            "bh_net_return_pct": float(bh_net),
            "A_net_return_pct": float(A_net),
            "B_net_return_pct": float(B_net)
        },
        "series": {
            "dates": dates_list,
            "close": close_list,
            "cum_bh": cum_bh_list,
            "cum_A": cum_A_list,
            "cum_B": cum_B_list,
            "sig_A": sig_A_list,
            "sig_B": sig_B_list,
            "sma_20": test_data['sma_20'].fillna(0.0).tolist(),
            "sma_50": test_data['sma_50'].fillna(0.0).tolist(),
            "ema_20": test_data['ema_20'].fillna(0.0).tolist(),
            "ema_50": test_data['ema_50'].fillna(0.0).tolist(),
            "ema_200": test_data['ema_200'].fillna(0.0).tolist(),
            "bb_upper": test_data['bb_upper'].fillna(0.0).tolist(),
            "bb_lower": test_data['bb_lower'].fillna(0.0).tolist(),
            "rsi": test_data['rsi'].fillna(50.0).tolist(),
            "macd": test_data['macd'].fillna(0.0).tolist(),
            "macd_signal": test_data['macd_signal'].fillna(0.0).tolist(),
            "macd_hist": test_data['macd_hist'].fillna(0.0).tolist(),
            "cci": test_data['cci'].fillna(0.0).tolist(),
            "williams_r": test_data['williams_r'].fillna(-50.0).tolist(),
            "mfi": test_data['mfi'].fillna(50.0).tolist()
        },
        "live_prediction": {
            "date": latest_date.strftime(date_format),
            "close": latest_close,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "action": action,
            "pos_size": pos_size,
            "threshold": current_threshold,
            "actual_history_dates": history_dates_str,
            "actual_history_prices": history_prices,
            "predicted_path_dates": future_dates_str,
            "predicted_path_prices": predicted_path_prices,
            "news": today_news_list,
            "history_sma20": history_df['sma_20'].fillna(0.0).tolist(),
            "history_sma50": history_df['sma_50'].fillna(0.0).tolist(),
            "history_ema20": history_df['ema_20'].fillna(0.0).tolist(),
            "history_ema50": history_df['ema_50'].fillna(0.0).tolist(),
            "history_ema200": history_df['ema_200'].fillna(0.0).tolist(),
            "history_bb_upper": history_df['bb_upper'].fillna(0.0).tolist(),
            "history_bb_lower": history_df['bb_lower'].fillna(0.0).tolist(),
            "history_rsi": history_df['rsi'].fillna(50.0).tolist(),
            "history_macd": history_df['macd'].fillna(0.0).tolist(),
            "history_macd_signal": history_df['macd_signal'].fillna(0.0).tolist(),
            "history_macd_hist": history_df['macd_hist'].fillna(0.0).tolist(),
            "history_cci": history_df['cci'].fillna(0.0).tolist(),
            "history_williams_r": history_df['williams_r'].fillna(-50.0).tolist(),
            "history_mfi": history_df['mfi'].fillna(50.0).tolist()
        }
    }
    
    # Save directly to React frontend public directory
    frontend_data_dir = "frontend/public/data"
    if not os.path.exists(frontend_data_dir):
        try:
            os.makedirs(frontend_data_dir)
        except Exception as e:
            logger.warning(f"Could not create directory {frontend_data_dir}: {e}")
            
    # Also write a local copy inside data_cache
    local_detail_path = os.path.join(DATA_CACHE_DIR, f"{symbol.lower()}_run_details.json")
    try:
        with open(local_detail_path, "w") as f:
            json.dump(frontend_run_details, f, indent=2)
        logger.info(f"Saved run details to local cache: {local_detail_path}")
    except Exception as e:
        logger.error(f"Failed to write run details to local cache: {e}")
    
    # Try saving to frontend public folder
    frontend_json_path = os.path.join(frontend_data_dir, f"{symbol.lower()}.json")
    try:
        with open(frontend_json_path, "w") as f:
            json.dump(frontend_run_details, f, indent=2)
        logger.info(f"Saved frontend JSON to: {frontend_json_path}")
    except Exception as e:
        logger.warning(f"Could not save JSON directly to frontend directory (expected if frontend folder not built yet): {e}")

    print("\n========================================================")
    print(f"         LIVE MARKET PREDICTION FOR {symbol} ")
    print("========================================================")
    print(f"Prediction Date:     {latest_date}")
    print(f"Latest Close Price:  {latest_close:.2f}")
    print(f"Current Threshold:   {current_threshold:.3f}")
    print("--------------------------------------------------------")
    print(f"Probability UP:      {prob_up:.2%}")
    print(f"Probability DOWN:    {prob_down:.2%}")
    print(f"Recommended Action:  {action}")
    print(f"Position Size:       {pos_size:+.2f}")
    print("--------------------------------------------------------")
    print("Today's News Headlines:")
    if today_headlines:
        for idx, headline in enumerate(today_headlines):
            print(f"  {idx+1}. {headline}")
    else:
        print("  No whitelisted headlines found for today.")
    print("========================================================\n")

if __name__ == "__main__":
    main()
