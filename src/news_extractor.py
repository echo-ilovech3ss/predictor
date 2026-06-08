import os
import json
import time
from openai import OpenAI
from src.logger import logger

class NewsExtractor:
    """Handles structured extraction of sentiment and financial indicators from stock news."""
    
    def __init__(self):
        # Read OpenCode Zen settings from environment
        self.api_key = os.environ.get("OPENCODE_API_KEY", "sk-UTtWH65FGecBfTOPRyAywKm8grIOPhbqbZ2pmsjshsAzVD4XjqIg5QWdCcUzeQwW")
        self.base_url = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
        self.model = os.environ.get("OPENCODE_MODEL", "deepseek-v4-flash-free")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
    def extract_features_llm(self, title: str, retries: int = 3) -> dict:
        """
        Extract structured features from a news headline using the deepseek-v4-flash-free model.
        Returns a dict matching the specified output schema.
        """
        prompt = f"""Return ONLY JSON.

{{
  "sentiment": -1 to 1,
  "importance": 0 to 1,
  "bull_score": 0 to 10,
  "bear_score": 0 to 10,
  "risk_score": 0 to 10,

  "earnings": true/false,
  "guidance_change": true/false,
  "partnership": true/false,
  "lawsuit": true/false,
  "product_launch": true/false,
  "management_change": true/false
}}

Analyze this financial headline:
"{title}"
"""
        for attempt in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                content = completion.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                logger.warning(f"LLM extraction attempt {attempt + 1} failed for '{title}': {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"LLM extraction permanently failed: {e}")
                    raise e
                    
    def extract_features_heuristic(self, title: str) -> dict:
        """
        Heuristic extraction that mimics the LLM feature extraction logic.
        Parses keywords and rules to output the exact same schema.
        """
        title_lower = title.lower()
        
        # 1. Detect events
        earnings = any(w in title_lower for w in ["earning", "q1", "q2", "q3", "q4", "revenue", "profit", "results", "quarter"])
        guidance_change = any(w in title_lower for w in ["guidance", "outlook", "forecast", "estimates", "cut", "raise", "guidance"])
        partnership = any(w in title_lower for w in ["partner", "alliance", "collaborate", "team", "deal", "agreement"])
        lawsuit = any(w in title_lower for w in ["lawsuit", "sue", "suit", "litigation", "court", "antitrust", "probe", "investigate", "fine", "regulatory"])
        product_launch = any(w in title_lower for w in ["launch", "announce", "introduce", "unveil", "release", "new product", "vision pro", "iphone", "macbook", "ipad", "ai feature"])
        management_change = any(w in title_lower for w in ["ceo", "cfo", "resign", "appoint", "successor", "hire", "leave", "step down"])
        
        # 2. Heuristic Sentiment & Score calculation
        sentiment = 0.0
        importance = 0.3  # Default background importance
        
        # Bullish and Bearish lexicons
        bull_words = ["upgrade", "buy", "rise", "gain", "raise", "beat", "higher", "surge", "growth", "partnership", "success", "innovate", "partnership", "bullish", "record", "jump", "positive"]
        bear_words = ["downgrade", "sell", "fall", "lose", "cut", "miss", "lower", "slump", "decline", "lawsuit", "antitrust", "drop", "trim", "bearish", "plunge", "negative"]
        
        bull_count = sum(1 for w in bull_words if w in title_lower)
        bear_count = sum(1 for w in bear_words if w in title_lower)
        
        if bull_count > bear_count:
            sentiment = 0.15 * min(bull_count, 4)
            importance = 0.5 + 0.1 * min(bull_count, 4)
        elif bear_count > bull_count:
            sentiment = -0.15 * min(bear_count, 4)
            importance = 0.5 + 0.1 * min(bear_count, 4)
            
        # Specific event adjustments
        if lawsuit:
            sentiment -= 0.3
            importance = max(importance, 0.7)
        if partnership:
            sentiment += 0.2
            importance = max(importance, 0.5)
        if product_launch:
            sentiment += 0.2
            importance = max(importance, 0.6)
        if guidance_change:
            importance = max(importance, 0.8)
            if any(w in title_lower for w in ["raise", "lift", "upward", "boost"]):
                sentiment += 0.4
            elif any(w in title_lower for w in ["cut", "lower", "trim", "slashed"]):
                sentiment -= 0.4
                
        # Clip sentiment to [-1.0, 1.0]
        sentiment = max(-1.0, min(1.0, sentiment))
        
        # Bull/Bear/Risk scores out of 10
        bull_score = int(round((sentiment + 1.0) * 5.0))
        bear_score = 10 - bull_score
        
        # Risk score calculation
        risk_score = 2  # default base risk
        if lawsuit:
            risk_score += 4
        if guidance_change and sentiment < 0:
            risk_score += 3
        if bear_count > bull_count:
            risk_score += 2
        risk_score = max(0, min(10, risk_score))
        
        return {
            "sentiment": float(sentiment),
            "importance": float(importance),
            "bull_score": int(bull_score),
            "bear_score": int(bear_score),
            "risk_score": int(risk_score),
            "earnings": bool(earnings),
            "guidance_change": bool(guidance_change),
            "partnership": bool(partnership),
            "lawsuit": bool(lawsuit),
            "product_launch": bool(product_launch),
            "management_change": bool(management_change)
        }
