import argparse
import datetime
import sys
import pandas as pd
from config import Config
from src.logger import logger
from src.alerts import AlertSystem
from src.data_fetcher import YFinanceProvider
from src.indicators import calculate_indicators
from src.features import prepare_data_for_training, extract_features
from src.market_state import classify_market_states, classify_market_state_row
from src.ml_model import MarketMLModel
from src.strategy import Strategy
from src.backtester import Backtester
from src.paper_trader import PaperTrader
from src.costs import TAX_DISCLAIMER

def parse_args():
    parser = argparse.ArgumentParser(description="Paper-Trading Market Prediction Bot CLI")
    parser.add_argument(
        "action",
        choices=["train", "backtest", "predict"],
        help="Action to perform: 'train' model, run 'backtest' historically, or 'predict' the next state."
    )
    parser.add_argument(
        "--symbol",
        required=True,
        choices=["NIFTY", "SPY"],
        help="Market symbol to analyze: 'NIFTY' (NIFTY 50) or 'SPY' (S&P 500 ETF)."
    )
    return parser.parse_args()

def get_provider_and_dates(symbol: str, action: str):
    """Return configured data provider and start/end dates based on action."""
    provider = YFinanceProvider()
    
    # Configure dates based on action
    if action == "predict":
        # Fetch 60 days of 1-hour candles to ensure we have >200 candles for indicators (EMA200, MACD, etc.)
        start = datetime.datetime.now() - datetime.timedelta(days=60)
    else:
        # Train / Backtest: Fetch max allowable (729 days) to get a good history
        start = datetime.datetime.now() - datetime.timedelta(days=729)
        
    end = datetime.datetime.now()
    return provider, start, end

def run_train(symbol: str):
    logger.info(f"--- TRAINING MODE FOR {symbol} ---")
    provider, start, end = get_provider_and_dates(symbol, "train")
    
    # 1. Fetch data
    df = provider.fetch_data(symbol, start, end)
    if df.empty or AlertSystem.is_halted():
        logger.error("Training aborted: data fetch failed or system halted.")
        return
        
    # 2. Calculate indicators
    df_with_ind = calculate_indicators(df)
    if len(df_with_ind) < 200:
        logger.error("Not enough historical data points after calculation. Need > 200 candles.")
        return
        
    # 3. Extract features & target labels
    X, y = prepare_data_for_training(df_with_ind)
    
    # 4. Train Model
    ml_model = MarketMLModel(symbol)
    ml_model.train(X, y)
    logger.info(f"Model trained and saved successfully for {symbol}.")

def run_backtest(symbol: str):
    logger.info(f"--- BACKTEST MODE FOR {symbol} ---")
    provider, start, end = get_provider_and_dates(symbol, "backtest")
    
    # 1. Fetch data
    df = provider.fetch_data(symbol, start, end)
    if df.empty or AlertSystem.is_halted():
        logger.error("Backtest aborted: data fetch failed or system halted.")
        return
        
    # 2. Calculate indicators
    df = calculate_indicators(df)
    
    # 3. Assign market state to each bar
    df['market_state'] = classify_market_states(df)
    
    # 4. Try loading ML model
    ml_model = MarketMLModel(symbol)
    has_ml = ml_model.load()
    
    # 5. Run backtest with ML Strategy
    backtester = Backtester(df, symbol)
    
    if has_ml:
        logger.info("Running ML-guided strategy backtest...")
        ml_results = backtester.run_backtest(ml_model)
    else:
        logger.warning("No ML model found. Run 'train' first. Proceeding with Rule-only backtest...")
        ml_results = None
        
    # 6. Run Rule-only baseline backtest
    logger.info("Running Rule-only baseline backtest...")
    rule_results = backtester.run_backtest(ml_model=None)
    
    # 7. Print results comparison
    print("\n==================================================")
    print(f"       BACKTEST RESULTS COMPARISON: {symbol}")
    print("==================================================")
    print(TAX_DISCLAIMER)
    print("--------------------------------------------------")
    
    # Buy and Hold stats (same for both)
    bh_ret = rule_results.get('bh_return_after_costs_pct', 0)
    print(f"Buy and Hold Net Return:       {bh_ret:.2f}%")
    print(f"Buy and Hold Final Capital:    {rule_results.get('bh_final_capital', 0):.2f}")
    print("--------------------------------------------------")
    
    print("Rule-Only Baseline Strategy:")
    print(f"  Total Trades:                {rule_results.get('trade_count', 0)}")
    print(f"  Win Rate:                    {rule_results.get('win_rate', 0)*100:.2f}%")
    print(f"  Max Drawdown:                {rule_results.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Net Return:                  {rule_results.get('total_return_after_costs_pct', 0):.2f}%")
    print(f"  Final Capital:               {rule_results.get('final_capital_after_costs', 0):.2f}")
    
    if ml_results:
        print("--------------------------------------------------")
        print("ML-Guided Strategy:")
        print(f"  Total Trades:                {ml_results.get('trade_count', 0)}")
        print(f"  Win Rate:                    {ml_results.get('win_rate', 0)*100:.2f}%")
        print(f"  Max Drawdown:                {ml_results.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Net Return:                  {ml_results.get('total_return_after_costs_pct', 0):.2f}%")
        print(f"  Final Capital:               {ml_results.get('final_capital_after_costs', 0):.2f}")
        print(f"  Profit Factor:               {ml_results.get('profit_factor', 0):.2f}")
        
    print("==================================================")

def run_predict(symbol: str):
    logger.info(f"--- PREDICTION & PAPER TRADING EXECUTION FOR {symbol} ---")
    provider, start, end = get_provider_and_dates(symbol, "predict")
    
    # 1. Fetch recent data (e.g. last 30 days)
    df = provider.fetch_data(symbol, start, end)
    if df.empty or AlertSystem.is_halted():
        logger.error("Prediction aborted: data fetch failed or system halted.")
        return
        
    # 2. Calculate indicators
    df_with_ind = calculate_indicators(df)
    if len(df_with_ind) < 200:
        logger.error("Not enough historical data points to run prediction. Need > 200 candles.")
        return
        
    # Extract latest row
    latest_row = df_with_ind.iloc[-1]
    latest_time = df_with_ind.index[-1]
    latest_price = latest_row['close']
    
    # 3. Classify market state
    latest_row_dict = latest_row.to_dict()
    market_state = classify_market_state_row(latest_row_dict)
    latest_row_dict['market_state'] = market_state
    
    # 4. Load ML model and predict probabilities
    ml_model = MarketMLModel(symbol)
    if not ml_model.load():
        logger.error("No trained model found! Train the model first using 'train' action.")
        return
        
    features_df = extract_features(df_with_ind)
    latest_features = features_df.iloc[[-1]] # double brackets keep it as DataFrame
    
    # Get probabilities
    # ml_model.predict_proba returns [prob_down, prob_up]
    try:
        probas = ml_model.predict_proba(latest_features)
        prob_down, prob_up = probas[0], probas[1]
    except Exception as e:
        logger.error(f"Failed to generate model prediction probabilities: {e}")
        prob_down, prob_up = 0.5, 0.5
        
    # 5. Generate strategy action recommendation
    strat = Strategy()
    signal = strat.generate_signal(latest_row_dict, prob_up, prob_down)
    
    # 6. Update paper trading state
    paper_trader = PaperTrader(symbol)
    time_str = latest_time.strftime("%Y-%m-%d %H:%M:%S")
    paper_trader.process_signal(signal, latest_price, time_str)
    
    # 7. Print results
    print("\n==================================================")
    print(f"        LIVE PREDICTION SIGNAL: {symbol}")
    print("==================================================")
    print(f"Time (Local):        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Time:           {time_str}")
    print(f"Latest Close Price:  {latest_price:.2f}")
    print(f"Market State:        {market_state.upper()}")
    print("--------------------------------------------------")
    print(f"Probability UP:      {prob_up*100:.1f}%")
    print(f"Probability DOWN:    {prob_down*100:.1f}%")
    print(f"Confidence Score:    {signal['confidence']*100:.1f}%")
    print(f"Action:              {signal['action'].upper()}")
    print(f"Explanation:         {signal['explanation']}")
    print("--------------------------------------------------")
    print("Paper Trading Status:")
    print(f"  Cash Balance:      {paper_trader.cash:.2f}")
    if paper_trader.position:
        pos = paper_trader.position
        print(f"  Active Position:   {pos['qty']:.2f} shares entered at {pos['entry_price']:.2f}")
        print(f"  Current Value:     {(pos['qty'] * latest_price):.2f}")
        print(f"  Current PnL:       {((latest_price - pos['entry_price']) * pos['qty']):.2f}")
        print(f"  Stop Loss:         {pos['stop_loss']:.2f}")
    else:
        print("  Active Position:   NONE (Flat)")
    print(f"  Total Portfolio:   {paper_trader.get_equity(latest_price):.2f}")
    print("==================================================")

def main():
    args = parse_args()
    
    if args.action == "train":
        run_train(args.symbol)
    elif args.action == "backtest":
        run_backtest(args.symbol)
    elif args.action == "predict":
        run_predict(args.symbol)
        
if __name__ == "__main__":
    main()
