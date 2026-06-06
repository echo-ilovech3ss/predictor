import pandas as pd
from config import Config
from src.logger import logger

class Strategy:
    """Combines ML probabilities and technical rules to make BUY/SELL/HOLD recommendations."""
    
    def __init__(self, min_confidence: float = None):
        self.min_confidence = min_confidence if min_confidence is not None else Config.MIN_CONFIDENCE_FOR_TRADE
        
    def generate_signal(self, 
                        row_data: dict, 
                        prob_up: float, 
                        prob_down: float) -> dict:
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
        
        # Determine confidence
        confidence = max(prob_up, prob_down)
        
        action = "HOLD"
        explanation = ""
        
        # 1. Evaluate Entry (BUY) signal first
        # Enter if:
        # - Model probability of UP is high (prob_up >= min_confidence)
        # - Market state is bullish or sideways (do not buy in bearish, volatile, or uncertain states)
        # - Price is above EMA 50 (bullish filter)
        # - RSI is not overbought (< 70)
        if prob_up >= self.min_confidence:
            if market_state in ["bullish", "sideways"] and close > ema_50 and rsi < 70:
                action = "BUY"
                explanation = (
                    f"BUY recommended because: trend is {market_state}, "
                    f"price ({close:.2f}) is above EMA 50 ({ema_50:.2f}), "
                    f"RSI is healthy ({rsi:.1f}), and model estimates a "
                    f"{prob_up*100:.1f}% probability of upward movement."
                )
            else:
                # ML was bullish, but filter failed
                action = "HOLD"
                failed_filters = []
                if market_state not in ["bullish", "sideways"]:
                    failed_filters.append(f"market state is {market_state}")
                if close <= ema_50:
                    failed_filters.append(f"price ({close:.2f}) is below EMA 50 ({ema_50:.2f})")
                if rsi >= 70:
                    failed_filters.append(f"RSI is overbought ({rsi:.1f})")
                    
                explanation = (
                    f"HOLD: Model predicted upward movement with {prob_up*100:.1f}% confidence, "
                    f"but entry filters blocked trade because: {', and '.join(failed_filters)}."
                )
            
        # 2. Evaluate Exit (SELL) signal
        # Exit if:
        # - Model probability of DOWN is high (prob_down >= min_confidence)
        # - Market state is bearish
        # - RSI is extremely overbought (suggesting potential quick pullback, e.g. > 80)
        elif prob_down >= self.min_confidence or market_state == "bearish" or rsi > 80:
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
            
        # Safety/Config limit warnings
        if Config.PAPER_TRADING_ONLY is False:
            # Re-enforce safety
            action = "HOLD"
            explanation = "SYSTEM HALTED: PAPER_TRADING_ONLY must be set to true."
            
        return {
            "action": action,
            "confidence": confidence,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "market_state": market_state,
            "explanation": explanation
        }
