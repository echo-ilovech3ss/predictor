import pandas as pd
from config import Config
from src.logger import logger

class Strategy:
    """Combines ML probabilities and technical rules to make BUY/SELL/HOLD recommendations."""
    
    def __init__(self, min_confidence: float = None):
        self.min_confidence = min_confidence if min_confidence is not None else Config.MIN_CONFIDENCE_FOR_TRADE
        
    def generate_signal(self, 
                        row_data: dict, 
                        prob_up: float = None, 
                        prob_down: float = None,
                        use_ml: bool = True) -> dict:
        """
        Generate trading recommendations.
        Long-only strategy:
        - BUY: Open long.
        - SELL: Exit long.
        - HOLD: No action/remain flat.
        """
        market_state = row_data.get('market_state', 'uncertain')
        close = row_data.get('close', 0)
        ema_50 = row_data.get('ema_50', 0)
        rsi = row_data.get('rsi', 50)
        
        action = "HOLD"
        explanation = ""
        confidence = 0.50
        
        # Helper variables for rule checks
        tech_buy_conditions_met = (market_state == "bullish" and close > ema_50 and rsi < 70)
        tech_sell_conditions_met = (market_state == "bearish" or rsi > 80)
        
        if use_ml:
            if prob_up is None or prob_down is None:
                prob_up, prob_down = 0.5, 0.5
                
            confidence = max(prob_up, prob_down)
            
            # 1. Evaluate Entry (BUY) signal
            if prob_up >= self.min_confidence:
                if tech_buy_conditions_met:
                    action = "BUY"
                    explanation = (
                        f"BUY recommended because: trend is {market_state}, "
                        f"price ({close:.2f}) is above EMA 50 ({ema_50:.2f}), "
                        f"RSI is healthy ({rsi:.1f}), and model estimates a "
                        f"{prob_up*100:.1f}% probability of upward movement."
                    )
                else:
                    action = "HOLD"
                    failed_filters = []
                    if market_state != "bullish":
                        failed_filters.append(f"market state is {market_state} (must be bullish)")
                    if close <= ema_50:
                        failed_filters.append(f"price ({close:.2f}) is below EMA 50 ({ema_50:.2f})")
                    if rsi >= 70:
                        failed_filters.append(f"RSI is overbought ({rsi:.1f})")
                        
                    explanation = (
                        f"HOLD: Model predicted upward movement with {prob_up*100:.1f}% confidence, "
                        f"but entry filters blocked trade because: {', and '.join(failed_filters)}."
                    )
                
            # 2. Evaluate Exit (SELL) signal
            elif prob_down >= self.min_confidence or tech_sell_conditions_met:
                action = "SELL"
                reasons = []
                if prob_down >= self.min_confidence:
                    reasons.append(f"model estimates a high downward probability of {prob_down*100:.1f}%")
                if market_state == "bearish":
                    reasons.append("market state is bearish (under EMA 50 & 200)")
                if rsi > 80:
                    reasons.append(f"RSI is extremely overbought ({rsi:.1f})")
                explanation = f"SELL (Exit Long) recommended because: {', and '.join(reasons)}."
            
            # 3. Default to HOLD (confidence is low)
            else:
                action = "HOLD"
                explanation = (
                    f"HOLD recommended: Model confidence (UP: {prob_up*100:.1f}%, DOWN: {prob_down*100:.1f}%) "
                    f"is below the threshold of {self.min_confidence*100:.1f}%."
                )
        else:
            # Rule-Only Baseline: Ignore ML probabilities and confidence entirely
            if tech_buy_conditions_met:
                action = "BUY"
                explanation = (
                    f"BUY recommended (Rule-Only): trend is {market_state}, "
                    f"price ({close:.2f}) is above EMA 50 ({ema_50:.2f}), "
                    f"and RSI is healthy ({rsi:.1f})."
                )
            elif tech_sell_conditions_met:
                action = "SELL"
                reasons = []
                if market_state == "bearish":
                    reasons.append("market state is bearish (under EMA 50 & 200)")
                if rsi > 80:
                    reasons.append(f"RSI is extremely overbought ({rsi:.1f})")
                explanation = f"SELL (Rule-Only Exit Long) recommended because: {', and '.join(reasons)}."
            else:
                action = "HOLD"
                explanation = "HOLD recommended (Rule-Only): Technical entry/exit rules are not met."
                
        # Safety/Config limit warnings
        if Config.PAPER_TRADING_ONLY is False:
            action = "HOLD"
            explanation = "SYSTEM HALTED: PAPER_TRADING_ONLY must be set to true."
            
        return {
            "action": action,
            "confidence": confidence,
            "prob_up": prob_up if use_ml else 0.50,
            "prob_down": prob_down if use_ml else 0.50,
            "market_state": market_state,
            "explanation": explanation
        }
