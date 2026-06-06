# AlphaPredict: Market Prediction & Paper-Trading Bot

AlphaPredict is a modular, cautious, and realistic paper-trading and backtesting engine for quantitative market prediction. It analyzes 1-hour OHLCV candle data for the **S&P 500 (SPY)** and **NIFTY 50 (India)**.

> [!IMPORTANT]
> **FINANCIAL & LEGAL DISCLAIMER**  
> This software is strictly for **educational, research, and simulation purposes**. It contains **no brokerage integration or live order execution capabilities** and **never trades real money**. It does not guarantee profits, and all outputs are probabilistic estimates. Tax and transaction cost calculations are simplified assumptions and do not constitute formal financial, legal, or tax advice.

---

## What the Bot Does

1.  **Fetches & Caches Market Data**: Connects to Yahoo Finance to fetch 1-hour candles. It maintains a local CSV cache to overcome the standard `yfinance` 730-day history limit.
2.  **Calculates Technical Indicators**: Vectorized calculations for EMA (20, 50, 200), RSI (14), MACD, Bollinger Bands, ATR, rolling volatility, and volume dynamics.
3.  **Classifies Market States**: Groups market conditions into `bullish`, `bearish`, `sideways`, `volatile`, or `uncertain`.
4.  **Calibrated ML Engine**: Trains a `RandomForestClassifier` with `CalibratedClassifierCV` to predict next-candle direction ($Close_{t+1} > Close_t$) using time-series splits and strict anti-leakage scaling.
5.  **Long-Only Risk Strategy**: Generates BUY (open long), SELL (exit long), and HOLD signals under strict position sizing, stop-loss (2%), and daily loss limit (5%) checks.
6.  **Next-Bar Execution Backtester**: Simulates trades realistically by generating signals at the close of candle $t$ and executing entries/exits at the Open price of candle $t+1$.
7.  **Sleek Streamlit Dashboard**: Provides an interactive, dark-themed UI to visualize metrics, interactive Plotly charts, backtest statistics, and paper-trading portfolio logs.

## What the Bot Does NOT Do

*   **No Live Trading**: Does not place real orders or connect to live broker APIs.
*   **No Short Selling**: Strictly executes long-only trades. A SELL signal represents an exit from an active long position, not opening a short position.
*   **No Guarantees**: Does not claim or promise profitability. Default action is HOLD when model confidence is low.

---

## Installation

Ensure you have Python 3.9+ installed.

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment**:
    Copy `.env.example` to `.env` (it is created automatically with defaults):
    ```bash
    cp .env.example .env
    ```

---

## Configuration (`.env`)

The `.env` file exposes the following parameters:
*   `STARTING_CAPITAL`: Starting simulated balance (default: `100000.0`).
*   `STOP_LOSS_PCT`: Stop loss percentage per trade (default: `2.0` for 2%).
*   `DAILY_LOSS_LIMIT_PCT`: Daily loss threshold before trading halts (default: `5.0` for 5%).
*   `MAX_POSITION_SIZE_PCT`: Max capital percentage per trade (default: `20.0` for 20%).
*   `MAX_TRADES_PER_DAY`: Max trades allowed in a single day (default: `3`).
*   `MIN_CONFIDENCE_FOR_TRADE`: Min probability threshold for BUY/SELL actions (default: `0.60` for 60%).
*   `SLIPPAGE_PCT`: Slippage cost per trade (default: `0.05` for 0.05%).
*   `BROKERAGE_PCT`: Brokerage commission rate (default: `0.03` for 0.03%).
*   `TAX_ON_PROFIT_PCT`: Simplified capital gains tax on profitable trades (default: `15.0` for 15%).
*   `PAPER_TRADING_ONLY`: Safety toggle (must remain `true`).

---

## Usage Instructions

### 1. Train the ML Model
Train the calibrated random forest classifier on historical data:
```bash
# Train for S&P 500
python main.py train --symbol SPY

# Train for NIFTY 50
python main.py train --symbol NIFTY
```

### 2. Run Backtest
Run backtest over the last 730 days, comparing the ML-guided strategy against Buy & Hold and a Rule-only baseline:
```bash
# Backtest SPY
python main.py backtest --symbol SPY

# Backtest NIFTY
python main.py backtest --symbol NIFTY
```

### 3. Generate Predictions & Step Paper Trader
Fetch the latest candles, make predictions, and execute paper trades on the local database (`spy_paper_trader_state.json` or `nifty_paper_trader_state.json`):
```bash
# Predict & trade SPY
python main.py predict --symbol SPY

# Predict & trade NIFTY
python main.py predict --symbol NIFTY
```

### 4. Run Streamlit Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```

---

## Understanding Signals

*   **BUY**: Recommendation to enter a long position. Triggered if model probability of upward movement is $\ge$ `MIN_CONFIDENCE_FOR_TRADE`, market state is `bullish` or `sideways`, price is above the 50 EMA, and RSI is not overbought.
*   **SELL**: Signal to exit an active long position. Triggered if the model estimates a downward probability $\ge$ `MIN_CONFIDENCE_FOR_TRADE`, if the market state becomes `bearish`, or if RSI exceeds 80.
*   **HOLD**: Default action. Triggered when model confidence is low, risk limits are breached, or technical indicators are conflicting.

---

## Fee & Tax Assumptions

*   **Slippage**: Simulated as a percentage penalty added to entry price (buying higher) and subtracted from exit price (selling lower).
*   **US Fees**: Standard brokerage + minor SEC and FINRA regulatory fees on sell orders.
*   **Indian Fees**: Brokerage + Securities Transaction Tax (STT, 0.1% buy/sell) + GST (18% on brokerage) + Stamp Duty (0.015% buy).
*   **Taxes**: Levied only on profitable completed trades (net of fees). India STCG is capped at a minimum of 20%, whereas the US STCG uses the `.env` default (15%).
