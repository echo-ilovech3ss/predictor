import os
from abc import ABC, abstractmethod
import datetime
import pandas as pd
import pytz
import yfinance as yf
from src.logger import logger
from src.alerts import AlertSystem

class BaseDataProvider(ABC):
    """Abstract Base Class for Market Data Providers."""
    
    @abstractmethod
    def fetch_data(self, symbol: str, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
        """Fetch historical candle data (OHLCV)."""
        pass

class YFinanceProvider(BaseDataProvider):
    """Yahoo Finance Data Provider with local caching and timezone validation."""
    
    # Define timezones and market hour configurations for NIFTY and SPY
    MARKET_CONFIGS = {
        "NIFTY": {
            "symbol": "^NSEI",
            "timezone": "Asia/Kolkata",
            "market_start": datetime.time(9, 15),
            "market_end": datetime.time(15, 30),
            "cache_file": "nifty_1h.csv"
        },
        "SPY": {
            "symbol": "SPY",
            "timezone": "America/New_York",
            "market_start": datetime.time(9, 30),
            "market_end": datetime.time(16, 0),
            "cache_file": "spy_1h.csv"
        }
    }
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
    def _get_config_key(self, symbol: str) -> str:
        """Resolve generic symbols to config keys."""
        s_upper = symbol.upper()
        if "NIFTY" in s_upper or "^NSEI" in s_upper:
            return "NIFTY"
        return "SPY"  # Default to SPY/US
        
    def fetch_data(self, symbol: str, start: datetime.datetime = None, end: datetime.datetime = None) -> pd.DataFrame:
        """
        Fetch 1-hour OHLCV data using yfinance. 
        Uses local CSV caching to accumulate historical data beyond yfinance's 730-day limit.
        """
        config_key = self._get_config_key(symbol)
        config = self.MARKET_CONFIGS[config_key]
        yf_symbol = config["symbol"]
        tz = pytz.timezone(config["timezone"])
        
        # Local cache path
        cache_path = os.path.join(self.cache_dir, config["cache_file"])
        
        # Determine timeframe dates in target timezone
        now_tz = datetime.datetime.now(tz)
        if end is None:
            end = now_tz
        if start is None:
            # yfinance allows max 730 days for 1h candles in a single request
            start = now_tz - datetime.timedelta(days=729)
            
        # Ensure timezone awareness
        if start.tzinfo is None:
            start = tz.localize(start)
        if end.tzinfo is None:
            end = tz.localize(end)
            
        logger.info(f"Requested data for {symbol} ({yf_symbol}) from {start} to {end}")
        
        # Load cache if it exists
        cached_df = pd.DataFrame()
        if os.path.exists(cache_path):
            try:
                cached_df = pd.read_csv(cache_path, index_col=0)
                if not cached_df.empty:
                    # Parse index robustly as localized datetime
                    cached_df.index = pd.to_datetime(cached_df.index, utc=True).tz_convert(tz)
                    logger.info(f"Loaded {len(cached_df)} rows of cached data for {symbol}.")
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_path}: {e}. Will overwrite cache.")
                
        # If we have cache, we only fetch what is missing (between the last cache timestamp and 'end')
        if not cached_df.empty:
            last_cached_time = cached_df.index[-1]
            # If the last cached time is close to 'end', we don't need to fetch much
            if last_cached_time < end - datetime.timedelta(hours=1):
                fetch_start = last_cached_time - datetime.timedelta(hours=2) # Fetch with overlap to be safe
                logger.info(f"Cache is stale. Fetching updates from {fetch_start} to {end}...")
                new_df = self._download_from_yfinance(yf_symbol, fetch_start, end)
                if not new_df.empty:
                    # Standardize index
                    new_df.index = pd.to_datetime(new_df.index).tz_convert(tz)
                    # Combine cache and new data
                    combined_df = pd.concat([cached_df, new_df]).sort_index()
                    # Drop duplicate timestamps, keeping the newest records
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    cached_df = combined_df
                    # Save back to cache
                    cached_df.to_csv(cache_path)
                    logger.info(f"Cache updated and saved. Total rows: {len(cached_df)}.")
            df = cached_df
        else:
            # No cache, fetch full requested range (limited to 729 days by yfinance if it's 1h)
            logger.info(f"No cache found. Fetching clean data for {symbol}...")
            df = self._download_from_yfinance(yf_symbol, start, end)
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_convert(tz)
                df.to_csv(cache_path)
                logger.info(f"Saved initial fetch of {len(df)} rows to cache.")
                
        if df.empty:
            AlertSystem.trigger_alert(f"Failed to fetch any data for symbol: {symbol}", halt_system=True)
            return pd.DataFrame()
            
        # Filter data for requested window
        filtered_df = df.loc[start:end]
        
        # Clean columns and index names
        filtered_df = filtered_df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        })
        # Check standard columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in filtered_df.columns:
                AlertSystem.trigger_alert(f"Missing required column {col} in fetched data for {symbol}", halt_system=True)
                return pd.DataFrame()
                
        # Drop rows with NaN close prices
        filtered_df = filtered_df.dropna(subset=["close"])
        
        # Check if we need to merge cross-market features (NIFTY only)
        if config_key == "NIFTY":
            try:
                logger.info("NIFTY symbol detected: Merging overnight SPY cross-market features...")
                # Fetch SPY data recursively using UTC zone alignment (start 2 days earlier to avoid start boundary NaNs)
                spy_df = self.fetch_data("SPY", start - datetime.timedelta(days=2), end)
                if not spy_df.empty:
                    # Calculate SPY returns
                    spy_df['spy_returns'] = spy_df['close'].pct_change().fillna(0)
                    spy_df = spy_df.rename(columns={'close': 'spy_close'})
                    
                    # Align timezones via UTC
                    nifty_utc = filtered_df.tz_convert(pytz.UTC)
                    spy_utc = spy_df.tz_convert(pytz.UTC)
                    
                    # Merge based on matching previous SPY candle
                    merged_utc = pd.merge_asof(
                        nifty_utc, 
                        spy_utc[['spy_close', 'spy_returns']], 
                        left_index=True, 
                        right_index=True, 
                        direction='backward'
                    )
                    
                    # Convert back to NIFTY standard timezone
                    filtered_df = merged_utc.tz_convert(tz)
                    # Forward-fill any NaN values and fill remaining with 0
                    filtered_df['spy_close'] = filtered_df['spy_close'].ffill().fillna(0)
                    filtered_df['spy_returns'] = filtered_df['spy_returns'].ffill().fillna(0)
                    logger.info("Successfully merged SPY cross-market features with NIFTY.")
            except Exception as e:
                logger.error(f"Failed to merge cross-market SPY features: {e}")
        
        # Validate data recency
        self.validate_recency(filtered_df, symbol)
        
        return filtered_df
        
    def _download_from_yfinance(self, symbol: str, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
        """Download raw data from yfinance API directly."""
        try:
            # yfinance uses string dates or datetime objects
            # Download with 1h interval
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval="1h")
            if df.empty:
                # Fallback to history with period if start is too far back
                logger.warning(f"yfinance returned empty for {symbol} in dates {start} to {end}. Trying max period...")
                df = ticker.history(period="730d", interval="1h")
            return df
        except Exception as e:
            AlertSystem.trigger_alert(f"yfinance API error for {symbol}: {e}", halt_system=True)
            return pd.DataFrame()
            
    def validate_recency(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate if the data is recent based on the market timezone and trading hours.
        Weekends, holidays, and overnight gaps should not trigger a data failure.
        """
        if df.empty:
            AlertSystem.trigger_alert(f"Validation failed: empty DataFrame for {symbol}", halt_system=True)
            return False
            
        config_key = self._get_config_key(symbol)
        config = self.MARKET_CONFIGS[config_key]
        tz = pytz.timezone(config["timezone"])
        
        # Current time in symbol's timezone
        now_tz = datetime.datetime.now(tz)
        
        # Last candle timestamp in localized DataFrame
        last_timestamp = df.index[-1]
        if last_timestamp.tzinfo is None:
            last_timestamp = tz.localize(last_timestamp)
        else:
            last_timestamp = last_timestamp.tz_convert(tz)
            
        logger.info(f"Recency check for {symbol}: Last candle timestamp is {last_timestamp}. Current time is {now_tz}.")
        
        # Check if the gap is unacceptable (e.g. > 96 hours to allow for long holidays/weekends)
        time_gap = now_tz - last_timestamp
        
        # Calculate expected gap based on weekend or holiday
        # If it's Saturday or Sunday, we expect a larger gap
        max_acceptable_hours = 96 if now_tz.weekday() in (5, 6) else 24
        
        if time_gap > datetime.timedelta(hours=max_acceptable_hours):
            # Stale data alert
            msg = f"Data for {symbol} is stale. Last candle is {last_timestamp} (gap of {time_gap.total_seconds() / 3600:.1f} hours)."
            AlertSystem.trigger_alert(msg, level="WARNING", halt_system=False)
            return False
            
        logger.info(f"Recency check passed for {symbol}.")
        return True
