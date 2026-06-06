import argparse
import datetime
import sys
import pandas as pd
import numpy as np
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
        choices=["train", "backtest", "predict", "walk-forward"],
        help="Action to perform: 'train' model, run 'backtest' historically, 'predict' the next state, or run 'walk-forward' validation."
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
        # Train / Backtest / Walk-Forward: Fetch max allowable (729 days) to get a good history
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
    
    # 1. Fetch data (is_live=False to bypass recency check)
    df = provider.fetch_data(symbol, start, end, is_live=False)
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
    
    # Calculate chronological test start date (last 20% of dataset)
    split_idx = int(len(df) * 0.8)
    test_start_date = df.index[split_idx]
    
    # 5. Run backtest with ML Strategy
    backtester = Backtester(df, symbol)
    
    if has_ml:
        logger.info(f"Running ML-guided strategy backtest (OOS Start: {test_start_date.date()})...")
        ml_results = backtester.run_backtest(ml_model, test_start_date=test_start_date)
    else:
        logger.warning("No ML model found. Run 'train' first. Proceeding with Rule-only backtest...")
        ml_results = None
        
    # 6. Run Rule-only baseline backtest
    logger.info("Running Rule-only baseline backtest...")
    rule_results = backtester.run_backtest(ml_model=None, test_start_date=test_start_date)
    
    # 7. Print results comparison
    print("\n==================================================")
    print(f"       BACKTEST RESULTS COMPARISON: {symbol}")
    print("==================================================")
    print(TAX_DISCLAIMER)
    print("--------------------------------------------------")
    
    # Buy and Hold stats (same for both)
    bh_is = rule_results.get('bh_is_return_pct', 0)
    bh_oos = rule_results.get('bh_oos_return_pct', 0)
    print(f"Buy and Hold Net Return (In-Sample):   {bh_is:.2f}%")
    print(f"Buy and Hold Net Return (Out-of-Sample): {bh_oos:.2f}%")
    print("--------------------------------------------------")
    
    # Rule Only IS/OOS
    rule_is = rule_results.get('is_metrics', {})
    rule_oos = rule_results.get('oos_metrics', {})
    print("Rule-Only Baseline Strategy:")
    print("  [In-Sample Period]")
    print(f"    Trades Count:              {rule_is.get('trade_count', 0)}")
    if rule_is.get('warning_msg'):
        print(f"    {rule_is.get('warning_msg')}")
    print(f"    Win Rate:                  {rule_is.get('win_rate', 0)*100:.2f}%")
    print(f"    Max Drawdown:              {rule_is.get('max_drawdown_pct', 0):.2f}%")
    print(f"    Net Return:                {rule_is.get('total_return_pct', 0):.2f}%")
    print("  [Out-of-Sample Period]")
    print(f"    Trades Count:              {rule_oos.get('trade_count', 0)}")
    if rule_oos.get('warning_msg'):
        print(f"    {rule_oos.get('warning_msg')}")
    print(f"    Win Rate:                  {rule_oos.get('win_rate', 0)*100:.2f}%")
    print(f"    Max Drawdown:              {rule_oos.get('max_drawdown_pct', 0):.2f}%")
    print(f"    Net Return:                {rule_oos.get('total_return_pct', 0):.2f}%")
    
    if ml_results:
        ml_is = ml_results.get('is_metrics', {})
        ml_oos = ml_results.get('oos_metrics', {})
        print("--------------------------------------------------")
        print("ML-Guided Strategy:")
        print("  [In-Sample Period]")
        print(f"    Trades Count:              {ml_is.get('trade_count', 0)}")
        if ml_is.get('warning_msg'):
            print(f"    {ml_is.get('warning_msg')}")
        print(f"    Win Rate:                  {ml_is.get('win_rate', 0)*100:.2f}%")
        print(f"    Max Drawdown:              {ml_is.get('max_drawdown_pct', 0):.2f}%")
        print(f"    Net Return:                {ml_is.get('total_return_pct', 0):.2f}%")
        print(f"    Profit Factor:             {ml_is.get('profit_factor', 0):.2f}")
        print(f"    Avg Win:                   ${ml_is.get('avg_win_cash', 0):.2f} ({ml_is.get('avg_win_pct', 0):+.2f}%)")
        print(f"    Avg Loss:                  ${ml_is.get('avg_loss_cash', 0):.2f} ({ml_is.get('avg_loss_pct', 0):+.2f}%)")
        print("  [Out-of-Sample Period]")
        print(f"    Trades Count:              {ml_oos.get('trade_count', 0)}")
        if ml_oos.get('warning_msg'):
            print(f"    {ml_oos.get('warning_msg')}")
        print(f"    Win Rate:                  {ml_oos.get('win_rate', 0)*100:.2f}%")
        print(f"    Max Drawdown:              {ml_oos.get('max_drawdown_pct', 0):.2f}%")
        print(f"    Net Return:                {ml_oos.get('total_return_pct', 0):.2f}%")
        print(f"    Profit Factor:             {ml_oos.get('profit_factor', 0):.2f}")
        print(f"    Avg Win:                   ${ml_oos.get('avg_win_cash', 0):.2f} ({ml_oos.get('avg_win_pct', 0):+.2f}%)")
        print(f"    Avg Loss:                  ${ml_oos.get('avg_loss_cash', 0):.2f} ({ml_oos.get('avg_loss_pct', 0):+.2f}%)")
        
    print("==================================================")

def run_predict(symbol: str):
    logger.info(f"--- PREDICTION & PAPER TRADING EXECUTION FOR {symbol} ---")
    provider, start, end = get_provider_and_dates(symbol, "predict")
    
    # 1. Fetch recent data (is_live=True to check recency)
    df = provider.fetch_data(symbol, start, end, is_live=True)
    if df.empty or AlertSystem.is_halted():
        logger.error("Prediction aborted: data fetch failed or system halted.")
        return
        
    # Check market open/stale status
    is_open, is_stale, status_msg = provider.check_market_status(symbol, df)
    
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
    try:
        probas = ml_model.predict_proba(latest_features)
        prob_down, prob_up = probas[0], probas[1]
    except Exception as e:
        logger.error(f"Failed to generate model prediction probabilities: {e}")
        prob_down, prob_up = 0.5, 0.5
        
    # 5. Generate strategy action recommendation
    strat = Strategy(min_confidence=ml_model.optimal_threshold)
    
    if not is_open or is_stale:
        signal = {
            "action": "HOLD",
            "confidence": 0.50,
            "prob_up": 0.50,
            "prob_down": 0.50,
            "market_state": market_state,
            "explanation": f"HOLD: Prediction forced to HOLD because market is closed or data is stale ({status_msg})."
        }
    else:
        signal = strat.generate_signal(latest_row_dict, prob_up, prob_down)
    
    # 6. Update paper trading state
    paper_trader = PaperTrader(symbol)
    time_str = latest_time.strftime("%Y-%m-%d %H:%M:%S")
    paper_trader.process_signal(signal, latest_row_dict, time_str, is_market_open=is_open, is_data_stale=is_stale)
    
    # 7. Print results
    print("\n==================================================")
    print(f"        LIVE PREDICTION SIGNAL: {symbol}")
    print("==================================================")
    print(f"Time (Local):        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Time:           {time_str}")
    print(f"Latest Close Price:  {latest_price:.2f}")
    print(f"Market State:        {market_state.upper()}")
    print("--------------------------------------------------")
    print(f"Market Status:       {status_msg}")
    print(f"Probability UP:      {prob_up*100:.1f}%")
    print(f"Probability DOWN:    {prob_down*100:.1f}%")
    print(f"Confidence Score:    {signal['confidence']*100:.1f}%")
    print(f"Action:              {signal['action'].upper()}")
    print(f"Explanation:         {signal['explanation']}")
    print("--------------------------------------------------")
    print("Paper Trading Status:")
    print(f"  Cash Balance:      {paper_trader.cash:.2f}")
    if paper_trader.queued_signal:
        print(f"  Queued Signal:     {paper_trader.queued_signal} (Executes at next candle open)")
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

def run_walk_forward(symbol: str):
    logger.info(f"--- WALK-FORWARD VALIDATION FOR {symbol} ---")
    provider, start, end = get_provider_and_dates(symbol, "walk-forward")
    
    # Fetch data (is_live=False to bypass recency check during historical simulation)
    df = provider.fetch_data(symbol, start, end, is_live=False)
    if df.empty:
        logger.error("Walk-forward validation aborted: data fetch failed.")
        return
        
    # Calculate indicators
    df = calculate_indicators(df)
    df['market_state'] = classify_market_states(df)
    
    start_date = df.index[0]
    end_date = df.index[-1]
    
    train_duration = datetime.timedelta(days=365)
    test_duration = datetime.timedelta(days=60)
    
    train_end = start_date + train_duration
    
    oos_probs_up = []
    oos_probs_down = []
    oos_thresholds = []
    oos_index = []
    
    walk_idx = 1
    while train_end + test_duration <= end_date:
        test_end = train_end + test_duration
        
        # Slice datasets
        df_train = df.loc[start_date:train_end]
        df_test = df.loc[train_end + pd.Timedelta(seconds=1):test_end]
        
        # Skip if test window is empty
        if len(df_test) == 0 or len(df_train) < 200:
            train_end += test_duration
            continue
            
        logger.info(f"Walk {walk_idx}: Training from {df_train.index[0].date()} to {df_train.index[-1].date()}, Testing from {df_test.index[0].date()} to {df_test.index[-1].date()}")
        
        try:
            # Prepare features & labels on df_train
            X_train, y_train = prepare_data_for_training(df_train)
            
            # Extract features on df_test (OOS)
            X_test = extract_features(df_test)
            feature_names = list(X_train.columns)
            X_test = X_test[feature_names].fillna(0)
            
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.preprocessing import StandardScaler
            from xgboost import XGBClassifier
            from sklearn.metrics import recall_score, precision_score
            
            # Fit final scaler on X_train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # TimeSeriesSplit CV on X_train for threshold tuning
            tscv = TimeSeriesSplit(n_splits=5)
            val_preds_list = []
            val_targets_list = []
            
            for tr_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
                
                fold_scaler = StandardScaler()
                X_tr_sc = fold_scaler.fit_transform(X_tr)
                X_val_sc = fold_scaler.transform(X_val)
                
                fold_clf = XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.03,
                    min_child_weight=15, subsample=0.75, colsample_bytree=0.75,
                    reg_alpha=0.5, reg_lambda=5.0, scale_pos_weight=1.0,
                    random_state=42, n_jobs=-1, eval_metric='logloss'
                )
                fold_clf.fit(X_tr_sc, y_tr)
                val_preds_list.extend(fold_clf.predict_proba(X_val_sc)[:, 1])
                val_targets_list.extend(y_val)
                
            val_preds = np.array(val_preds_list)
            val_targets = np.array(val_targets_list)
            
            best_t = 0.50
            best_prec = 0.0
            for t in np.linspace(0.50, 0.65, 31):
                preds = (val_preds >= t).astype(int)
                rec = recall_score(val_targets, preds, zero_division=0)
                prec = precision_score(val_targets, preds, zero_division=0)
                if rec >= 0.10 and prec > best_prec:
                    best_prec = prec
                    best_t = t
                    
            optimal_threshold = float(best_t)
            
            # Train final model on full X_train (past window only)
            clf = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.03,
                min_child_weight=15, subsample=0.75, colsample_bytree=0.75,
                reg_alpha=0.5, reg_lambda=5.0, scale_pos_weight=1.0,
                random_state=42, n_jobs=-1, eval_metric='logloss'
            )
            clf.fit(X_train_scaled, y_train)
            
            # Predict only on X_test (next OOS window)
            probas = clf.predict_proba(X_test_scaled)
            prob_down = probas[:, 0]
            prob_up = probas[:, 1]
            
            # Collect OOS predictions
            oos_probs_up.extend(prob_up)
            oos_probs_down.extend(prob_down)
            oos_thresholds.extend([optimal_threshold] * len(df_test))
            oos_index.extend(df_test.index)
            
        except Exception as e:
            logger.error(f"Error in Walk {walk_idx}: {e}", exc_info=True)
            
        # Slide training boundary forward
        train_end += test_duration
        walk_idx += 1
        
    if not oos_index:
        logger.error("No walk-forward periods completed successfully.")
        return
        
    # Concatenate OOS predictions
    pred_prob_up = pd.Series(oos_probs_up, index=oos_index)
    pred_prob_down = pd.Series(oos_probs_down, index=oos_index)
    dynamic_thresholds = pd.Series(oos_thresholds, index=oos_index)
    
    # Drop duplicates
    pred_prob_up = pred_prob_up[~pred_prob_up.index.duplicated(keep='last')].sort_index()
    pred_prob_down = pred_prob_down[~pred_prob_down.index.duplicated(keep='last')].sort_index()
    dynamic_thresholds = dynamic_thresholds[~dynamic_thresholds.index.duplicated(keep='last')].sort_index()
    
    first_oos_time = oos_index[0]
    oos_start_idx = df.index.get_loc(first_oos_time)
    
    # Backtest with padding index for indicators
    last_oos_time = oos_index[-1]
    oos_end_idx = df.index.get_loc(last_oos_time)
    df_backtest = df.iloc[max(0, oos_start_idx - 200) : oos_end_idx + 1]
    backtester = Backtester(df_backtest, symbol)
    
    # ML Strategy Backtest using concatenated OOS predictions and thresholds
    logger.info(f"Running ML-guided Walk-Forward backtest starting at {first_oos_time.date()}...")
    ml_results = backtester.run_backtest(
        ml_model=None,
        test_start_date=first_oos_time,
        pred_prob_up=pred_prob_up,
        pred_prob_down=pred_prob_down,
        dynamic_thresholds=dynamic_thresholds
    )
    
    # Rule-only baseline backtest
    logger.info("Running Rule-only baseline backtest...")
    rule_results = backtester.run_backtest(ml_model=None, test_start_date=first_oos_time)
    
    # Print results comparison
    print("\n==================================================")
    print(f"   WALK-FORWARD VALIDATION RESULTS: {symbol}")
    print("==================================================")
    print(f"OOS Backtest Period: {first_oos_time.date()} to {oos_index[-1].date()}")
    print("--------------------------------------------------")
    
    oos_bh = ml_results.get('bh_oos_return_pct', 0)
    print(f"Buy and Hold Net Return:       {oos_bh:.2f}%")
    print("--------------------------------------------------")
    
    rule_oos = rule_results.get('oos_metrics', {})
    print("Rule-Only Baseline Strategy (OOS):")
    print(f"  Total Trades:                {rule_oos.get('trade_count', 0)}")
    if rule_oos.get('warning_msg'):
        print(f"  {rule_oos.get('warning_msg')}")
    print(f"  Win Rate:                    {rule_oos.get('win_rate', 0)*100:.2f}%")
    print(f"  Max Drawdown:                {rule_oos.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Net Return:                  {rule_oos.get('total_return_pct', 0):.2f}%")
    
    print("--------------------------------------------------")
    
    ml_oos = ml_results.get('oos_metrics', {})
    print("ML-Guided Walk-Forward Strategy (OOS):")
    print(f"  Total Trades:                {ml_oos.get('trade_count', 0)}")
    if ml_oos.get('warning_msg'):
        print(f"  {ml_oos.get('warning_msg')}")
    print(f"  Win Rate:                    {ml_oos.get('win_rate', 0)*100:.2f}%")
    print(f"  Max Drawdown:                {ml_oos.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Net Return:                  {ml_oos.get('total_return_pct', 0):.2f}%")
    print(f"  Profit Factor:               {ml_oos.get('profit_factor', 0):.2f}")
    print(f"  Avg Win:                     ${ml_oos.get('avg_win_cash', 0):.2f} ({ml_oos.get('avg_win_pct', 0):+.2f}%)")
    print(f"  Avg Loss:                    ${ml_oos.get('avg_loss_cash', 0):.2f} ({ml_oos.get('avg_loss_pct', 0):+.2f}%)")
    print("==================================================")
    
    return ml_results, rule_results

def main():
    args = parse_args()
    
    if args.action == "train":
        run_train(args.symbol)
    elif args.action == "backtest":
        run_backtest(args.symbol)
    elif args.action == "predict":
        run_predict(args.symbol)
    elif args.action == "walk-forward":
        run_walk_forward(args.symbol)

if __name__ == "__main__":
    main()
