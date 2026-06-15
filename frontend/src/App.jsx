import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  RefreshCw, 
  BarChart2, 
  Calendar, 
  Newspaper, 
  ArrowRight, 
  Percent, 
  Activity, 
  Code,
  Info,
  CheckCircle,
  Copy,
  AlertCircle,
  Cpu
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine
} from 'recharts';
import './App.css';

const DEFAULT_SYMBOLS = [
  { value: 'AAPL', label: 'Apple Inc. (AAPL)' },
  { value: 'NIFTY', label: 'Nifty 50 Index (NIFTY)' },
  { value: 'NVDA', label: 'NVIDIA Corp. (NVDA)' },
  { value: 'TSLA', label: 'Tesla Inc. (TSLA)' },
  { value: 'MSFT', label: 'Microsoft Corp. (MSFT)' },
  { value: 'SPY', label: 'S&P 500 ETF (SPY)' }
];

const INTERVAL_OPTIONS = [
  { value: '1d', label: 'Daily (Swing/Invest)' },
  { value: '1wk', label: 'Weekly (Long-term)' },
  { value: '1h', label: '1 Hour (Swing)' },
  { value: '30m', label: '30 Minutes (Day Trade)' },
  { value: '15m', label: '15 Minutes (Day Trade)' },
  { value: '5m', label: '5 Minutes (Scalping)' }
];

const QUALITY_OPTIONS = [
  { value: '2', label: 'Quick (2 trials, ~5s)' },
  { value: '10', label: 'Standard (10 trials, ~25s)' },
  { value: '30', label: 'Deep Search (30 trials, ~1.5m)' }
];

export default function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [customSymbolInput, setCustomSymbolInput] = useState('');
  const [isCustomMode, setIsCustomMode] = useState(false);
  const [activeTab, setActiveTab] = useState('forecast');
  
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  
  // Sidebar config options for training
  const [horizonInput, setHorizonInput] = useState(5);
  const [intervalInput, setIntervalInput] = useState('1d');
  const [tuningQuality, setTuningQuality] = useState('2');

  // Retraining API states
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [trainingError, setTrainingError] = useState(null);

  // Technical Indicators States
  const [showIndicators, setShowIndicators] = useState(false);
  const [showSma20, setShowSma20] = useState(false);
  const [showSma50, setShowSma50] = useState(false);
  const [showEma20, setShowEma20] = useState(false);
  const [showEma50, setShowEma50] = useState(false);
  const [showEma200, setShowEma200] = useState(false);
  const [showBb, setShowBb] = useState(false);
  const [selectedOscillator, setSelectedOscillator] = useState('none');

  // Determine active target symbol based on custom input or select dropdown
  const targetSymbol = isCustomMode ? (customSymbolInput.trim().toUpperCase() || 'CUSTOM') : selectedSymbol;

  const isIndian = targetSymbol.toLowerCase().includes('nifty') || targetSymbol.toLowerCase().includes('nsei') || targetSymbol.toLowerCase().endsWith('.ns');
  const currencySymbol = isIndian ? '₹' : '$';

  const loadData = async (symbol, interval) => {
    if (!symbol) return;
    setLoading(true);
    setError(null);
    try {
      const activeInterval = interval || intervalInput;
      let response;
      
      if (interval) {
        // Explicit interval requested
        response = await fetch(`/data/${symbol.toLowerCase()}_${activeInterval.toLowerCase()}.json?t=${Date.now()}`);
        if (!response.ok) {
          response = await fetch(`/data/${symbol.toLowerCase()}.json?t=${Date.now()}`);
        }
      } else {
        // Initial symbol load - try main file first
        response = await fetch(`/data/${symbol.toLowerCase()}.json?t=${Date.now()}`);
        if (!response.ok) {
          response = await fetch(`/data/${symbol.toLowerCase()}_${activeInterval.toLowerCase()}.json?t=${Date.now()}`);
        }
      }
      
      if (!response.ok) {
        throw new Error('NOT_FOUND');
      }
      // Check if response is HTML (Vite fallback) instead of JSON
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('text/html')) {
        throw new Error('NOT_FOUND');
      }
      const jsonData = await response.json();
      
      // If specific interval requested, enforce it
      if (interval && jsonData.interval !== activeInterval) {
        throw new Error('NOT_FOUND');
      }
      
      setData(jsonData);
      // Synchronize input fields with loaded settings
      setHorizonInput(jsonData.horizon || 5);
      setIntervalInput(jsonData.interval || '1d');
    } catch (err) {
      console.error(err);
      setError(err.message === 'NOT_FOUND' ? 'NOT_FOUND' : 'FAILED_TO_LOAD');
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData(targetSymbol, null);
  }, [targetSymbol]);

  // Call the Vite proxy API to retrain, with optional custom parameters
  const handleStartTraining = async (optSymbol, optHorizon, optInterval) => {
    const symbol = (typeof optSymbol === 'string') ? optSymbol : targetSymbol;
    const horizon = (typeof optHorizon === 'number' || typeof optHorizon === 'string') ? optHorizon : horizonInput;
    const interval = (typeof optInterval === 'string') ? optInterval : intervalInput;

    setIsTraining(true);
    setTrainingError(null);
    setTrainingStatus(`Spawning backend optimizer for ${symbol} (${interval})...`);
    
    try {
      const query = `/api/train?symbol=${symbol}&horizon=${horizon}&interval=${interval}&trials=${tuningQuality}&no_walkforward=true`;
      const response = await fetch(query);
      const result = await response.json();
      
      if (result.success) {
        setTrainingStatus('Training successful! Reloading cache...');
        await loadData(symbol, interval);
      } else {
        throw new Error(result.error || 'Training failed');
      }
    } catch (err) {
      console.error(err);
      setTrainingError(err.message);
    } finally {
      setIsTraining(false);
      setTrainingStatus('');
    }
  };

  const handleIntervalChange = (newInterval) => {
    setIntervalInput(newInterval);
    loadData(targetSymbol, newInterval);
  };

  const handleCopyCommand = (command) => {
    navigator.clipboard.writeText(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Helper to format values
  const formatPercent = (val) => (val !== undefined ? `${(val * 100).toFixed(2)}%` : 'N/A');
  const formatValue = (val) => (val !== undefined && val !== null ? val.toFixed(2) : 'N/A');

  // Helper to get formatted hold period text
  const getHoldPeriodText = (horizon, interval) => {
    if (!horizon) return '';
    const unit = interval || '1d';
    switch (unit) {
      case '1d': return `${horizon} day${horizon > 1 ? 's' : ''}`;
      case '1wk': return `${horizon} week${horizon > 1 ? 's' : ''}`;
      case '1h': return `${horizon} hour${horizon > 1 ? 's' : ''}`;
      case '30m': return `${horizon * 30} mins`;
      case '15m': return `${horizon * 15} mins`;
      case '5m': return `${horizon * 5} mins`;
      default: return `${horizon} periods`;
    }
  };

  // Check if loaded model prediction cache is outdated
  const isCacheOutdated = (jsonData) => {
    const checkData = jsonData || data;
    if (!checkData || !checkData.live_prediction) return false;
    
    let predictionTimeUtc;
    const pred = checkData.live_prediction;
    
    if (pred.date_utc) {
      // Direct UTC timestamp available from backend (robust and timezone-offset immune)
      predictionTimeUtc = new Date(pred.date_utc);
    } else {
      // Fallback naive parsing if date_utc is missing
      const dateStr = pred.date;
      if (dateStr.includes(' ')) {
        predictionTimeUtc = new Date(dateStr.replace(' ', 'T') + ':00Z');
      } else {
        predictionTimeUtc = new Date(dateStr + 'T23:59:59Z');
      }
    }
    
    if (isNaN(predictionTimeUtc.getTime())) return false;
    
    const nowUtc = new Date();
    const diffMs = nowUtc - predictionTimeUtc;
    const dateStr = pred.date;
    
    // Warning thresholds: 1 hour for intraday, 24 hours for daily/weekly
    if (dateStr.includes(' ')) {
      return diffMs > 3600000;
    } else {
      return diffMs > 86400000;
    }
  };

  // Prepare data for the Trajectory Chart (Past 15 days + H days forecast)
  const getTrajectoryData = () => {
    if (!data || !data.live_prediction) return [];
    const chartData = [];
    const pred = data.live_prediction;

    // Add actual history
    pred.actual_history_dates.forEach((date, i) => {
      chartData.push({
        date: date,
        actual: pred.actual_history_prices[i],
        predicted: null,
        sma_20: pred.history_sma20 ? pred.history_sma20[i] : null,
        sma_50: pred.history_sma50 ? pred.history_sma50[i] : null,
        ema_20: pred.history_ema20 ? pred.history_ema20[i] : null,
        ema_50: pred.history_ema50 ? pred.history_ema50[i] : null,
        ema_200: pred.history_ema200 ? pred.history_ema200[i] : null,
        bb_upper: pred.history_bb_upper ? pred.history_bb_upper[i] : null,
        bb_lower: pred.history_bb_lower ? pred.history_bb_lower[i] : null,
        rsi: pred.history_rsi ? pred.history_rsi[i] : null,
        macd: pred.history_macd ? pred.history_macd[i] : null,
        macd_signal: pred.history_macd_signal ? pred.history_macd_signal[i] : null,
        macd_hist: pred.history_macd_hist ? pred.history_macd_hist[i] : null,
        cci: pred.history_cci ? pred.history_cci[i] : null,
        williams_r: pred.history_williams_r ? pred.history_williams_r[i] : null,
        mfi: pred.history_mfi ? pred.history_mfi[i] : null
      });
    });

    // Add future predictions (connecting first pred value to the last actual value)
    const lastHistIndex = chartData.length - 1;
    
    let r_sma20 = lastHistIndex >= 0 ? chartData[lastHistIndex].sma_20 : null;
    let r_sma50 = lastHistIndex >= 0 ? chartData[lastHistIndex].sma_50 : null;
    let r_ema20 = lastHistIndex >= 0 ? chartData[lastHistIndex].ema_20 : null;
    let r_ema50 = lastHistIndex >= 0 ? chartData[lastHistIndex].ema_50 : null;
    let r_ema200 = lastHistIndex >= 0 ? chartData[lastHistIndex].ema_200 : null;
    let r_bb_upper = lastHistIndex >= 0 ? chartData[lastHistIndex].bb_upper : null;
    let r_bb_lower = lastHistIndex >= 0 ? chartData[lastHistIndex].bb_lower : null;
    
    let r_rsi = lastHistIndex >= 0 ? chartData[lastHistIndex].rsi : null;
    let r_mfi = lastHistIndex >= 0 ? chartData[lastHistIndex].mfi : null;
    let r_williams_r = lastHistIndex >= 0 ? chartData[lastHistIndex].williams_r : null;
    let r_cci = lastHistIndex >= 0 ? chartData[lastHistIndex].cci : null;
    let r_macd = lastHistIndex >= 0 ? chartData[lastHistIndex].macd : null;
    let r_macd_signal = lastHistIndex >= 0 ? chartData[lastHistIndex].macd_signal : null;
    let r_macd_hist = lastHistIndex >= 0 ? chartData[lastHistIndex].macd_hist : null;
    
    const bb_width = (r_bb_upper !== null && r_bb_lower !== null) ? (r_bb_upper - r_bb_lower) : 0;

    pred.predicted_path_dates.forEach((date, i) => {
      const price = pred.predicted_path_prices[i];
      if (i === 0) {
        if (lastHistIndex >= 0) {
          chartData[lastHistIndex].predicted = price;
        }
      } else {
        if (price !== null && price !== undefined) {
          if (r_ema20 !== null) r_ema20 = price * (2 / 21) + r_ema20 * (1 - 2 / 21);
          if (r_ema50 !== null) r_ema50 = price * (2 / 51) + r_ema50 * (1 - 2 / 51);
          if (r_ema200 !== null) r_ema200 = price * (2 / 201) + r_ema200 * (1 - 2 / 201);
          if (r_sma20 !== null) r_sma20 = price * (2 / 21) + r_sma20 * (1 - 2 / 21);
          if (r_sma50 !== null) r_sma50 = price * (2 / 51) + r_sma50 * (1 - 2 / 51);
          
          if (r_sma20 !== null && bb_width > 0) {
            r_bb_upper = r_sma20 + bb_width / 2;
            r_bb_lower = r_sma20 - bb_width / 2;
          }
          
          if (r_rsi !== null) r_rsi = r_rsi * 0.85 + 50 * 0.15;
          if (r_mfi !== null) r_mfi = r_mfi * 0.85 + 50 * 0.15;
          if (r_williams_r !== null) r_williams_r = r_williams_r * 0.85 - 50 * 0.15;
          if (r_cci !== null) r_cci = r_cci * 0.85;
          if (r_macd !== null) r_macd = r_macd * 0.85;
          if (r_macd_signal !== null) r_macd_signal = r_macd_signal * 0.85;
          r_macd_hist = (r_macd !== null && r_macd_signal !== null) ? (r_macd - r_macd_signal) : null;
        }
        
        chartData.push({
          date: date,
          actual: null,
          predicted: price,
          sma_20: r_sma20,
          sma_50: r_sma50,
          ema_20: r_ema20,
          ema_50: r_ema50,
          ema_200: r_ema200,
          bb_upper: r_bb_upper,
          bb_lower: r_bb_lower,
          rsi: r_rsi,
          mfi: r_mfi,
          williams_r: r_williams_r,
          cci: r_cci,
          macd: r_macd,
          macd_signal: r_macd_signal,
          macd_hist: r_macd_hist
        });
      }
    });

    return chartData;
  };

  // Prepare data for the Backtest Returns Chart
  const getBacktestData = () => {
    if (!data || !data.series) return [];
    return data.series.dates.map((date, i) => ({
      date: date,
      'Buy & Hold': data.series.cum_bh[i] * 100,
      'Model A (Tech Only)': data.series.cum_A[i] * 100,
      'Model B (Tech + News)': data.series.cum_B[i] * 100
    }));
  };

  // Build the copyable run command
  const cliCommand = `python run_mvp.py --symbol ${targetSymbol} --horizon ${horizonInput} --interval ${intervalInput} --tuning-trials ${tuningQuality} --no-walkforward`;

  return (
    <div className="app-container">
      <div className="glow-backdrop-1"></div>
      <div className="glow-backdrop-2"></div>

      {/* Header */}
      <header className="app-header">
        <div className="logo-section">
          <Activity size={24} className="logo-icon" />
          <span className="logo-text">TRADE PREDICTOR</span>
          <span className="logo-tag">MODEL B COCKPIT</span>
        </div>
        <div className="header-meta" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {data && (
            <button 
              className="btn-secondary"
              onClick={() => handleStartTraining()}
              disabled={isTraining}
              style={{ padding: '6px 12px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}
            >
              <RefreshCw size={12} className={isTraining ? 'loader' : ''} style={{ animation: isTraining ? 'spin 1s linear infinite' : 'none' }} />
              Refresh Data & Model
            </button>
          )}
          <span className="legend-item" style={{ fontSize: '13px' }}>
            <span className="legend-color" style={{ backgroundColor: '#10b981' }}></span> Live System Ready
          </span>
        </div>
      </header>

      {/* Content Body */}
      <main className="content-wrapper">
        
        {/* Sidebar Controls */}
        <aside className="sidebar">
          <div className="glass-panel">
            <h3 className="sidebar-title">Select Ticker</h3>
            
            <div className="form-group">
              <label className="form-label">Ticker Mode</label>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
                <button 
                  className={`btn-secondary ${!isCustomMode ? 'active' : ''}`}
                  disabled={isTraining}
                  onClick={() => setIsCustomMode(false)}
                  style={{ flex: 1, padding: '6px' }}
                >
                  Presets
                </button>
                <button 
                  className={`btn-secondary ${isCustomMode ? 'active' : ''}`}
                  disabled={isTraining}
                  onClick={() => setIsCustomMode(true)}
                  style={{ flex: 1, padding: '6px' }}
                >
                  Custom Ticker
                </button>
              </div>
            </div>

            {!isCustomMode ? (
              <div className="form-group">
                <label className="form-label">Preconfigured Stocks</label>
                <select 
                  className="select-input"
                  value={selectedSymbol}
                  disabled={isTraining}
                  onChange={(e) => setSelectedSymbol(e.target.value)}
                >
                  {DEFAULT_SYMBOLS.map((s) => (
                    <option key={s.value} value={s.value}>{s.label}</option>
                  ))}
                </select>
              </div>
            ) : (
              <div className="form-group">
                <label className="form-label">Custom Yahoo Symbol</label>
                <input 
                  type="text" 
                  className="text-input"
                  placeholder="e.g. AMZN, GOOGL, RELIANCE.NS"
                  value={customSymbolInput}
                  disabled={isTraining}
                  onChange={(e) => setCustomSymbolInput(e.target.value)}
                />
                <span className="comparison-subtext" style={{ marginTop: '4px' }}>
                  Use suffix <code>.NS</code> for Indian stocks listed on NSE.
                </span>
              </div>
            )}

            <div className="form-group" style={{ borderTop: '1px solid var(--panel-border)', paddingTop: '16px' }}>
              <label className="form-label">Timeframe (Interval)</label>
              <select 
                className="select-input"
                value={intervalInput}
                disabled={isTraining}
                onChange={(e) => handleIntervalChange(e.target.value)}
              >
                {INTERVAL_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Set Prediction Horizon (Candles)</label>
              <input 
                type="range" 
                min="1" 
                max="10" 
                className="select-input" 
                style={{ padding: '0', cursor: 'pointer' }}
                value={horizonInput}
                disabled={isTraining}
                onChange={(e) => setHorizonInput(parseInt(e.target.value))}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--c-text-secondary)' }}>
                <span>1 Candle</span>
                <span style={{ fontWeight: '700', color: 'var(--c-accent)' }}>
                  {getHoldPeriodText(horizonInput, intervalInput)}
                </span>
                <span>10 Candles</span>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Bayesian Tuning Quality</label>
              <select 
                className="select-input"
                value={tuningQuality}
                disabled={isTraining}
                onChange={(e) => setTuningQuality(e.target.value)}
              >
                {QUALITY_OPTIONS.map((q) => (
                  <option key={q.value} value={q.value}>{q.label}</option>
                ))}
              </select>
            </div>

            <div style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <button 
                className="btn-primary"
                onClick={handleStartTraining}
                disabled={isTraining || (isCustomMode && !customSymbolInput)}
              >
                {isTraining ? <Cpu size={16} className="loader" style={{ animation: 'spin 1s linear infinite' }} /> : <Cpu size={16} />}
                {isTraining ? 'Training Model...' : 'Train Model in GUI'}
              </button>

              <button 
                className="btn-secondary"
                onClick={() => handleCopyCommand(cliCommand)}
                disabled={isTraining}
                style={{ display: 'flex', justifyContent: 'center' }}
              >
                {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
                {copied ? 'Copied Command!' : 'Copy CLI Command'}
              </button>
            </div>
          </div>

          <div className="glass-panel" style={{ marginTop: '16px', textAlign: 'left' }}>
            <h3 className="sidebar-title" style={{ margin: '0 0 12px 0' }}>Trader Tools</h3>
            
            <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <input 
                type="checkbox" 
                id="toggle-indicators" 
                checked={showIndicators} 
                onChange={(e) => setShowIndicators(e.target.checked)}
                style={{ cursor: 'pointer', width: '16px', height: '16px' }}
              />
              <label htmlFor="toggle-indicators" className="form-label" style={{ margin: 0, cursor: 'pointer', fontWeight: 'bold' }}>
                Enable Advanced Analysis
              </label>
            </div>
            
            {showIndicators && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', paddingLeft: '8px', borderLeft: '2px solid var(--c-accent-border)' }}>
                <h4 style={{ margin: '8px 0 4px 0', fontSize: '11px', textTransform: 'uppercase', color: 'var(--c-text-secondary)' }}>Overlays (On Price)</h4>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <input type="checkbox" id="show-sma20" checked={showSma20} onChange={(e) => setShowSma20(e.target.checked)} />
                  <label htmlFor="show-sma20" style={{ fontSize: '12px', cursor: 'pointer' }}>SMA (20)</label>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <input type="checkbox" id="show-sma50" checked={showSma50} onChange={(e) => setShowSma50(e.target.checked)} />
                  <label htmlFor="show-sma50" style={{ fontSize: '12px', cursor: 'pointer' }}>SMA (50)</label>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <input type="checkbox" id="show-ema20" checked={showEma20} onChange={(e) => setShowEma20(e.target.checked)} />
                  <label htmlFor="show-ema20" style={{ fontSize: '12px', cursor: 'pointer' }}>EMA (20)</label>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <input type="checkbox" id="show-ema50" checked={showEma50} onChange={(e) => setShowEma50(e.target.checked)} />
                  <label htmlFor="show-ema50" style={{ fontSize: '12px', cursor: 'pointer' }}>EMA (50)</label>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <input type="checkbox" id="show-ema200" checked={showEma200} onChange={(e) => setShowEma200(e.target.checked)} />
                  <label htmlFor="show-ema200" style={{ fontSize: '12px', cursor: 'pointer' }}>EMA (200)</label>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <input type="checkbox" id="show-bb" checked={showBb} onChange={(e) => setShowBb(e.target.checked)} />
                  <label htmlFor="show-bb" style={{ fontSize: '12px', cursor: 'pointer' }}>Bollinger Bands</label>
                </div>
                
                <h4 style={{ margin: '12px 0 4px 0', fontSize: '11px', textTransform: 'uppercase', color: 'var(--c-text-secondary)' }}>Oscillators (Sub-Charts)</h4>
                
                <select 
                  className="select-input"
                  value={selectedOscillator}
                  onChange={(e) => setSelectedOscillator(e.target.value)}
                  style={{ padding: '6px', fontSize: '12px', width: '100%' }}
                >
                  <option value="none">None</option>
                  <option value="rsi">RSI (Relative Strength)</option>
                  <option value="macd">MACD (Trend Momentum)</option>
                  <option value="cci">CCI (Commodity Channel)</option>
                  <option value="mfi">MFI (Money Flow Volume)</option>
                  <option value="williams_r">Williams %R</option>
                </select>
              </div>
            )}
          </div>

          <div className="glass-panel" style={{ fontSize: '12px', color: 'var(--c-text-secondary)' }}>
            <h4 style={{ margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--c-text-primary)' }}>
              <Info size={14} className="logo-icon" /> Timeframe Rules
            </h4>
            <p style={{ lineHeight: '140%', textAlign: 'left' }}>
              Short intraday timeframes (e.g. 5m, 15m) are heavily driven by trader momentum. The backend automatically <strong>damps daily news influence (by up to 80%)</strong> on these intervals, relying instead on candlestick patterns and indicators.
            </p>
          </div>
        </aside>

        {/* Main Dashboard Section */}
        <section className="main-panel">
          
          {isTraining && (
            <div className="glass-panel empty-container" style={{ border: '1px solid var(--c-accent-border)', background: 'rgba(99, 102, 241, 0.05)' }}>
              <Cpu size={48} className="loader" style={{ animation: 'spin 2s linear infinite', color: 'var(--c-accent)' }} />
              <h2 style={{ fontSize: '20px', margin: '8px 0' }}>Ensemble Model Optimizer Active</h2>
              <p style={{ color: 'var(--c-text-secondary)', maxWidth: '500px', fontSize: '14px', lineHeight: '145%' }}>
                {trainingStatus || 'Running Python script in the background. Running feature engineering and tuning hyperparameters...'}
              </p>
              <div className="probability-bar-container" style={{ maxWidth: '400px', height: '6px', marginTop: '16px' }}>
                <div className="probability-bar" style={{ width: '100%', background: 'linear-gradient(90deg, #6366f1, #10b981)', animation: 'spin 2s linear infinite' }}></div>
              </div>
            </div>
          )}

          {!isTraining && loading ? (
            <div className="glass-panel loading-container">
              <div className="loader"></div>
              <span>Fetching model forecasts for {targetSymbol}...</span>
            </div>
          ) : !isTraining && error === 'NOT_FOUND' ? (
            <div className="glass-panel empty-container">
              <Code size={48} style={{ color: 'var(--c-accent)' }} />
              <h2 style={{ fontSize: '20px', margin: '8px 0' }}>Dataset Cache Empty for "{targetSymbol}"</h2>
              <p style={{ color: 'var(--c-text-secondary)', maxWidth: '500px', fontSize: '14px', lineHeight: '145%' }}>
                We don't have cached training outcomes for **{targetSymbol}** on this interval. Click **Train Model in GUI** above to start training now.
              </p>
              
              <div className="terminal-box">
                {cliCommand}
              </div>

              <div style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
                <button 
                  className="btn-primary"
                  onClick={handleStartTraining}
                  style={{ width: 'auto', padding: '10px 20px' }}
                >
                  <Cpu size={16} /> Train Model Now
                </button>
                <button 
                  className="btn-secondary"
                  onClick={() => handleStartTraining(targetSymbol, horizonInput, intervalInput)}
                >
                  <RefreshCw size={14} /> Fetch & Train Model
                </button>
              </div>
            </div>
          ) : !isTraining && error ? (
            <div className="glass-panel empty-container">
              <AlertCircle size={48} style={{ color: 'var(--c-sell)' }} />
              <h2>Failed to Load Stock Data</h2>
              <p style={{ color: 'var(--c-text-secondary)' }}>
                {trainingError || 'An error occurred while loading files. Ensure the yfinance backend ran successfully.'}
              </p>
              <button className="btn-primary" onClick={() => loadData(targetSymbol, intervalInput)} style={{ width: 'auto', marginTop: '12px' }}>
                <RefreshCw size={14} /> Try Again
              </button>
            </div>
          ) : !isTraining && (
            <>
              {/* Warning Banner for Outdated Prediction Cache */}
              {isCacheOutdated() && (
                <div className="warning-banner glass-panel" style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '12px 16px',
                  marginBottom: '20px',
                  borderRadius: '12px',
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                  background: 'linear-gradient(90deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.02))',
                  color: '#f87171',
                  fontSize: '14px'
                }}>
                  <AlertCircle size={18} style={{ color: '#ef4444', flexShrink: 0 }} />
                  <div style={{ flexGrow: 1 }}>
                    <strong>Outdated Prediction Cache:</strong> Predictions are based on market data from <strong>{data.live_prediction.date}</strong>. Retrain the model in the GUI to fetch the latest market data.
                  </div>
                  <button 
                    onClick={handleStartTraining}
                    style={{
                      padding: '6px 12px',
                      background: 'rgba(239, 68, 68, 0.2)',
                      border: '1px solid rgba(239, 68, 68, 0.4)',
                      borderRadius: '6px',
                      color: '#fff',
                      cursor: 'pointer',
                      fontSize: '12px',
                      fontWeight: 500,
                      transition: 'all 0.2s',
                      whiteSpace: 'nowrap'
                    }}
                    onMouseOver={(e) => e.target.style.background = 'rgba(239, 68, 68, 0.3)'}
                    onMouseOut={(e) => e.target.style.background = 'rgba(239, 68, 68, 0.2)'}
                  >
                    Retrain Now
                  </button>
                </div>
              )}

              {/* Tab Navigation */}
              <nav className="tab-bar">
                <button 
                  className={`tab-btn ${activeTab === 'forecast' ? 'active' : ''}`}
                  onClick={() => setActiveTab('forecast')}
                >
                  <Calendar size={16} /> 🎯 Today's Target Forecast
                </button>
                <button 
                  className={`tab-btn ${activeTab === 'backtest' ? 'active' : ''}`}
                  onClick={() => setActiveTab('backtest')}
                >
                  <BarChart2 size={16} /> 📈 Backtest Performance
                </button>
              </nav>

              {/* Tab 1: Live Forecast Cockpit */}
              {activeTab === 'forecast' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                  
                  {/* Forecast Status Cards */}
                  <div className="prediction-summary-grid">
                    
                    {/* Action Card */}
                    <div className="glass-panel prediction-card">
                      <div>
                        <span className="card-label">Recommended Action</span>
                        <div className={`action-badge ${data.live_prediction.action.includes('BUY') ? 'buy' : data.live_prediction.action.includes('SELL') ? 'sell' : 'hold'}`}>
                          {data.live_prediction.action.includes('BUY') ? <TrendingUp size={32} /> : data.live_prediction.action.includes('SELL') ? <TrendingDown size={32} /> : <Activity size={32} />}
                          
                          {/* Display the clean "BUY for <time_period>" text directly */}
                          {data.live_prediction.action.includes('BUY') 
                            ? `BUY for ${getHoldPeriodText(data.horizon, data.interval)}`
                            : data.live_prediction.action.includes('SELL')
                            ? `SELL for ${getHoldPeriodText(data.horizon, data.interval)}`
                            : 'HOLD'}
                        </div>
                      </div>

                      <div>
                        <div className="probability-bar-container">
                          <div 
                            className={`probability-bar ${data.live_prediction.action.includes('BUY') ? 'buy' : data.live_prediction.action.includes('SELL') ? 'sell' : 'hold'}`}
                            style={{ width: `${data.live_prediction.prob_up * 100}%` }}
                          ></div>
                        </div>
                        <div className="metric-row">
                          <span>Probability UP: <strong>{(data.live_prediction.prob_up * 100).toFixed(1)}%</strong></span>
                          <span>Probability DOWN: <strong>{(data.live_prediction.prob_down * 100).toFixed(1)}%</strong></span>
                        </div>
                      </div>
                    </div>

                    {/* Stats Grid */}
                    <div className="glass-panel stats-grid">
                      <div className="stat-item">
                        <span className="card-label">Prediction Date/Time</span>
                        <div className="stat-val" style={{ fontSize: '13px', marginTop: '10px' }}>{data.live_prediction.date}</div>
                      </div>
                      <div className="stat-item">
                        <span className="card-label">Latest Close</span>
                        <div className="stat-val">{currencySymbol}{formatValue(data.live_prediction.close)}</div>
                      </div>
                      <div className="stat-item">
                        <span className="card-label">Hold Duration</span>
                        <div className="stat-val" style={{ fontSize: '15px', marginTop: '8px' }}>
                          {data.live_prediction.action.includes('HOLD') ? 'N/A (No Position)' : getHoldPeriodText(data.horizon, data.interval)}
                        </div>
                      </div>
                      <div className="stat-item">
                        <span className="card-label">Recommended Size</span>
                        <div className="stat-val" style={{ color: data.live_prediction.pos_size > 0 ? 'var(--c-buy)' : data.live_prediction.pos_size < 0 ? 'var(--c-sell)' : 'var(--c-text-primary)' }}>
                          {data.live_prediction.pos_size > 0 ? `+${data.live_prediction.pos_size.toFixed(2)}` : data.live_prediction.pos_size.toFixed(2)}
                        </div>
                      </div>
                    </div>

                  </div>

                  {/* Multi-Step Path Trajectory Chart */}
                  <div className="glass-panel">
                    <div className="chart-title-section">
                      <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 600 }}>{targetSymbol} Predicted Price Trajectory (+{getHoldPeriodText(data.horizon, data.interval)})</h3>
                      <div className="chart-legend">
                        <span className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: '#94a3b8' }}></span> Historical Price (15 Candles)
                        </span>
                        <span className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: data.live_prediction.predicted_path_prices[data.horizon] >= data.live_prediction.close ? 'var(--c-buy)' : 'var(--c-sell)', border: '1px dashed' }}></span> Predicted Trajectory
                        </span>
                      </div>
                    </div>

                    <div className="chart-container">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={getTrajectoryData()} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                          <XAxis 
                            dataKey="date" 
                            stroke="var(--c-text-muted)" 
                            fontSize={10} 
                            tickLine={false}
                          />
                          <YAxis 
                            stroke="var(--c-text-muted)" 
                            fontSize={11}
                            tickLine={false}
                            domain={['auto', 'auto']}
                          />
                          <Tooltip 
                            contentStyle={{ 
                              background: 'rgba(15, 15, 25, 0.9)', 
                              border: '1px solid var(--panel-border)',
                              borderRadius: '8px',
                              color: '#fff',
                              fontSize: '13px'
                            }}
                            formatter={(value, name) => {
                              if (name === 'actual') return [`${currencySymbol}${value.toFixed(2)}`, 'Close Price'];
                              if (name === 'predicted') return [`${currencySymbol}${value.toFixed(2)}`, 'Predicted'];
                              return [`${currencySymbol}${value.toFixed(2)}`, name];
                            }}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="actual" 
                            stroke="#94a3b8" 
                            strokeWidth={2.5} 
                            dot={false}
                            activeDot={{ r: 6 }}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="predicted" 
                            stroke={data.live_prediction.predicted_path_prices[data.horizon] >= data.live_prediction.close ? 'var(--c-buy)' : 'var(--c-sell)'} 
                            strokeWidth={2.5} 
                            strokeDasharray="5 5"
                            dot={{ r: 3 }}
                            activeDot={{ r: 6 }}
                          />
                          {showIndicators && showSma20 && (
                            <Line type="monotone" dataKey="sma_20" stroke="#0ea5e9" strokeWidth={1.5} dot={false} name="SMA (20)" />
                          )}
                          {showIndicators && showSma50 && (
                            <Line type="monotone" dataKey="sma_50" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="SMA (50)" />
                          )}
                          {showIndicators && showEma20 && (
                            <Line type="monotone" dataKey="ema_20" stroke="#a855f7" strokeWidth={1.5} dot={false} name="EMA (20)" />
                          )}
                          {showIndicators && showEma50 && (
                            <Line type="monotone" dataKey="ema_50" stroke="#ec4899" strokeWidth={1.5} dot={false} name="EMA (50)" />
                          )}
                          {showIndicators && showEma200 && (
                            <Line type="monotone" dataKey="ema_200" stroke="#ef4444" strokeWidth={1.5} dot={false} name="EMA (200)" />
                          )}
                          {showIndicators && showBb && (
                            <Line type="monotone" dataKey="bb_upper" stroke="#64748b" strokeWidth={1.2} strokeDasharray="4 4" dot={false} name="BB Upper" />
                          )}
                          {showIndicators && showBb && (
                            <Line type="monotone" dataKey="bb_lower" stroke="#64748b" strokeWidth={1.2} strokeDasharray="4 4" dot={false} name="BB Lower" />
                          )}
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>

                    {showIndicators && selectedOscillator !== 'none' && (
                      <div className="chart-container" style={{ height: '170px', marginTop: '16px', borderTop: '1px dashed var(--panel-border)', paddingTop: '16px' }}>
                        <h4 style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--c-text-secondary)', margin: '0 0 8px 0', textAlign: 'left', fontWeight: 'bold' }}>
                          {selectedOscillator.toUpperCase()} Oscillator
                        </h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={getTrajectoryData().filter(d => d.actual !== null)} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                            <XAxis dataKey="date" stroke="var(--c-text-muted)" fontSize={9} tickLine={false} />
                            <YAxis stroke="var(--c-text-muted)" fontSize={9} tickLine={false} domain={selectedOscillator === 'rsi' || selectedOscillator === 'mfi' ? [0, 100] : selectedOscillator === 'williams_r' ? [-100, 0] : ['auto', 'auto']} />
                            <Tooltip 
                              contentStyle={{ background: 'rgba(15, 15, 25, 0.9)', border: '1px solid var(--panel-border)', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                              formatter={(value) => [value.toFixed(2), '']}
                            />
                            {selectedOscillator === 'rsi' && (
                              <>
                                <Line type="monotone" dataKey="rsi" stroke="#10b981" strokeWidth={1.5} dot={false} />
                                <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" label={{ value: '70', fill: '#ef4444', fontSize: 8, position: 'right' }} />
                                <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" label={{ value: '30', fill: '#10b981', fontSize: 8, position: 'right' }} />
                              </>
                            )}
                            {selectedOscillator === 'mfi' && (
                              <>
                                <Line type="monotone" dataKey="mfi" stroke="#0ea5e9" strokeWidth={1.5} dot={false} />
                                <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="3 3" label={{ value: '80', fill: '#ef4444', fontSize: 8, position: 'right' }} />
                                <ReferenceLine y={20} stroke="#10b981" strokeDasharray="3 3" label={{ value: '20', fill: '#10b981', fontSize: 8, position: 'right' }} />
                              </>
                            )}
                            {selectedOscillator === 'williams_r' && (
                              <>
                                <Line type="monotone" dataKey="williams_r" stroke="#ec4899" strokeWidth={1.5} dot={false} />
                                <ReferenceLine y={-20} stroke="#ef4444" strokeDasharray="3 3" label={{ value: '-20', fill: '#ef4444', fontSize: 8, position: 'right' }} />
                                <ReferenceLine y={-80} stroke="#10b981" strokeDasharray="3 3" label={{ value: '-80', fill: '#10b981', fontSize: 8, position: 'right' }} />
                              </>
                            )}
                            {selectedOscillator === 'cci' && (
                              <>
                                <Line type="monotone" dataKey="cci" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                                <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="3 3" label={{ value: '100', fill: '#ef4444', fontSize: 8, position: 'right' }} />
                                <ReferenceLine y={-100} stroke="#10b981" strokeDasharray="3 3" label={{ value: '-100', fill: '#10b981', fontSize: 8, position: 'right' }} />
                              </>
                            )}
                            {selectedOscillator === 'macd' && (
                              <>
                                <Line type="monotone" dataKey="macd" stroke="#0ea5e9" strokeWidth={1.5} dot={false} name="MACD" />
                                <Line type="monotone" dataKey="macd_signal" stroke="#f59e0b" strokeWidth={1.2} dot={false} name="Signal" />
                              </>
                            )}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {/* Path Table */}
                    <div className="custom-table-container">
                      <table className="custom-table">
                        <thead>
                          <tr>
                            <th>Period</th>
                            <th>Date/Time</th>
                            <th>Predicted Stock Price</th>
                            <th>Cumulative Expected Change</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.live_prediction.predicted_path_dates.map((date, i) => {
                            const price = data.live_prediction.predicted_path_prices[i];
                            const basePrice = data.live_prediction.close;
                            const pctChange = ((price - basePrice) / basePrice) * 100;
                            return (
                              <tr key={i}>
                                <td style={{ fontWeight: 600 }}>
                                  {i === 0 ? 'Current (Actual)' : 
                                   (data.interval === '1d' ? `Day +${i}` : 
                                    data.interval === '1wk' ? `Week +${i}` : 
                                    data.interval === '1h' ? `Hour +${i}` : 
                                    `Candle +${i}`)}
                                </td>
                                <td>{date}</td>
                                <td style={{ fontWeight: 600 }}>{currencySymbol}{price.toFixed(2)}</td>
                                <td style={{ 
                                  color: pctChange > 0.01 ? 'var(--c-buy)' : pctChange < -0.01 ? 'var(--c-sell)' : 'var(--c-text-primary)',
                                  fontWeight: 600
                                }}>
                                  {pctChange > 0.01 ? `+${pctChange.toFixed(2)}%` : `${pctChange.toFixed(2)}%`}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Today's News Headlines Feed */}
                  <div className="glass-panel">
                    <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Newspaper size={18} className="logo-icon" /> Headlines Processed for Today's Sentiment
                    </h3>
                    
                    <div className="news-grid">
                      {data.live_prediction.news && data.live_prediction.news.length > 0 ? (
                        data.live_prediction.news.map((item, idx) => (
                          <div className="news-card" key={idx}>
                            <div className={`sentiment-dot ${item.sentiment > 0.15 ? 'bullish' : item.sentiment < -0.15 ? 'bearish' : 'neutral'}`}></div>
                            <div className="news-title">{item.title}</div>
                            <div className="news-meta">
                              <span className={`badge ${item.sentiment > 0.15 ? 'bullish' : item.sentiment < -0.15 ? 'bearish' : 'neutral'}`}>
                                {item.sentiment > 0.15 ? `Bullish (${item.sentiment.toFixed(2)})` : item.sentiment < -0.15 ? `Bearish (${item.sentiment.toFixed(2)})` : `Neutral (${item.sentiment.toFixed(2)})`}
                              </span>
                              <span>Importance: {item.importance.toFixed(0)}</span>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div style={{ padding: '20px', color: 'var(--c-text-secondary)', fontSize: '14px' }}>
                          No whitelisted headlines found for this date. Model relied primarily on technical indicators and calendar features.
                        </div>
                      )}
                    </div>
                  </div>

                </div>
              )}

              {/* Tab 2: Backtest Analysis */}
              {activeTab === 'backtest' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                  
                  {/* Returns Chart */}
                  <div className="glass-panel">
                    <div className="chart-title-section">
                      <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 600 }}>Cumulative Out-of-Sample Strategy Returns ({getHoldPeriodText(data.horizon, data.interval)} horizon)</h3>
                      <div className="chart-legend">
                        <span className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: '#94a3b8', border: '1px dashed' }}></span> Buy & Hold
                        </span>
                        <span className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: 'var(--c-sell)' }}></span> Model A (Tech Only)
                        </span>
                        <span className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: 'var(--c-buy)' }}></span> Model B (Tech + News)
                        </span>
                      </div>
                    </div>

                    <div className="chart-container">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={getBacktestData()} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                          <XAxis 
                            dataKey="date" 
                            stroke="var(--c-text-muted)" 
                            fontSize={10} 
                            tickLine={false}
                          />
                          <YAxis 
                            stroke="var(--c-text-muted)" 
                            fontSize={11}
                            tickLine={false}
                            formatter={(value) => `${value.toFixed(0)}%`}
                          />
                          <Tooltip 
                            contentStyle={{ 
                              background: 'rgba(15, 15, 25, 0.9)', 
                              border: '1px solid var(--panel-border)',
                              borderRadius: '8px',
                              color: '#fff',
                              fontSize: '13px'
                            }}
                            formatter={(value) => [`${value.toFixed(2)}%`, '']}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="Buy & Hold" 
                            stroke="#94a3b8" 
                            strokeWidth={1.5} 
                            strokeDasharray="4 4"
                            dot={false}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="Model A (Tech Only)" 
                            stroke="var(--c-sell)" 
                            strokeWidth={2.0} 
                            dot={false}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="Model B (Tech + News)" 
                            stroke="var(--c-buy)" 
                            strokeWidth={2.5} 
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Backtest Metrics Tables */}
                  <div className="backtest-metrics-grid">
                    
                    <div className="glass-panel metric-panel">
                      <span className="card-label">Out-of-Sample Net Return</span>
                      <div className="metric-comparison-value metric-comparison-val-b">
                        {formatPercent(data.metrics.B_net_return_pct / 100)}
                      </div>
                      <div className="comparison-subtext">
                        Model A: <strong>{formatPercent(data.metrics.A_net_return_pct / 100)}</strong> | Buy & Hold: <strong>{formatPercent(data.metrics.bh_net_return_pct / 100)}</strong>
                      </div>
                    </div>

                    <div className="glass-panel metric-panel">
                      <span className="card-label">Model Predict Accuracy</span>
                      <div className="metric-comparison-value" style={{ color: data.metrics.B_accuracy >= data.metrics.A_accuracy ? 'var(--c-buy)' : 'var(--c-sell)' }}>
                        {formatPercent(data.metrics.B_accuracy)}
                      </div>
                      <div className="comparison-subtext">
                        Model A: <strong>{formatPercent(data.metrics.A_accuracy)}</strong> (Sentiment Delta: <strong>{((data.metrics.B_accuracy - data.metrics.A_accuracy) * 100).toFixed(2)}%</strong>)
                      </div>
                    </div>

                    <div className="glass-panel metric-panel">
                      <span className="card-label">Risk-Adjusted Sharpe Ratio</span>
                      <div className="metric-comparison-value" style={{ color: data.metrics.B_sharpe >= 0 ? 'var(--c-buy)' : 'var(--c-sell)' }}>
                        {data.metrics.B_sharpe.toFixed(2)}
                      </div>
                      <div className="comparison-subtext">
                        Model A: <strong>{data.metrics.A_sharpe.toFixed(2)}</strong> | Max Drawdown (B): <strong>{data.metrics.B_max_dd.toFixed(1)}%</strong>
                      </div>
                    </div>

                  </div>

                  {/* Metrics Table */}
                  <div className="glass-panel">
                    <h3 style={{ margin: '0 0 16px 0', fontSize: '15px', fontWeight: 600 }}>Detailed Backtesting Comparison Table</h3>
                    <div className="custom-table-container">
                      <table className="custom-table">
                        <thead>
                          <tr>
                            <th>Backtest Metric</th>
                            <th>Buy & Hold Benchmark</th>
                            <th>Model A (Technical Only)</th>
                            <th>Model B (Technical + News)</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td style={{ fontWeight: 600 }}>Net Strategy Return</td>
                            <td>{formatPercent(data.metrics.bh_net_return_pct / 100)}</td>
                            <td>{formatPercent(data.metrics.A_net_return_pct / 100)}</td>
                            <td style={{ fontWeight: 700, color: 'var(--c-buy)' }}>{formatPercent(data.metrics.B_net_return_pct / 100)}</td>
                          </tr>
                          <tr>
                            <td style={{ fontWeight: 600 }}>Classification Accuracy</td>
                            <td>N/A</td>
                            <td>{formatPercent(data.metrics.A_accuracy)}</td>
                            <td style={{ fontWeight: 700, color: data.metrics.B_accuracy >= data.metrics.A_accuracy ? 'var(--c-buy)' : 'var(--c-sell)' }}>{formatPercent(data.metrics.B_accuracy)}</td>
                          </tr>
                          <tr>
                            <td style={{ fontWeight: 600 }}>Precision Score</td>
                            <td>N/A</td>
                            <td>{formatPercent(data.metrics.A_precision)}</td>
                            <td>{formatPercent(data.metrics.B_precision)}</td>
                          </tr>
                          <tr>
                            <td style={{ fontWeight: 600 }}>Recall Score</td>
                            <td>N/A</td>
                            <td>{formatPercent(data.metrics.A_recall)}</td>
                            <td>{formatPercent(data.metrics.B_recall)}</td>
                          </tr>
                          <tr>
                            <td style={{ fontWeight: 600 }}>Sharpe Ratio</td>
                            <td>N/A</td>
                            <td>{data.metrics.A_sharpe.toFixed(3)}</td>
                            <td style={{ fontWeight: 700, color: data.metrics.B_sharpe >= data.metrics.A_sharpe ? 'var(--c-buy)' : 'var(--c-sell)' }}>{data.metrics.B_sharpe.toFixed(3)}</td>
                          </tr>
                          <tr>
                            <td style={{ fontWeight: 600 }}>Max Drawdown (%)</td>
                            <td>N/A</td>
                            <td>{data.metrics.A_max_dd.toFixed(1)}%</td>
                            <td>{data.metrics.B_max_dd.toFixed(1)}%</td>
                          </tr>
                          <tr>
                            <td style={{ fontWeight: 600 }}>Total Trade Signals (Long/Short)</td>
                            <td>N/A</td>
                            <td>{data.metrics.A_longs_count} Longs / {data.metrics.A_shorts_count} Shorts</td>
                            <td>{data.metrics.B_longs_count} Longs / {data.metrics.B_shorts_count} Shorts</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                </div>
              )}

            </>
          )}

        </section>
      </main>
    </div>
  );
}
