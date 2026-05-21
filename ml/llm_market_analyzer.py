"""
LLM-Based Market Analyzer for DART v3.0
Uses state-of-the-art language models to analyze market sentiment, news,
and provide intelligent trading insights.
"""

import json
import logging
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("llm_market_analyzer")


class LLMMarketAnalyzer:
    """
    State-of-the-art LLM-based market analyzer using open-source models.
    Analyzes market data, sentiment, and news to provide trading insights.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        embedding_model: str = "all-MiniLM-L6-v2",
        base_url: str = "http://localhost:8000/v1",
        allow_fallback: bool = False,
    ):
        """
        Initialize LLM Market Analyzer.

        Args:
            model_name: Name of the LLM model (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')
            embedding_model: Sentence transformer model for embeddings
            base_url: Base URL for vLLM API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.allow_fallback = allow_fallback
        self.is_real_llm_available = False

        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key="EMPTY",
                openai_api_base=base_url,
                temperature=0.3,
                max_tokens=512,
            )
            self.is_real_llm_available = True
            logger.info(f"vLLM client initialized: {model_name} at {base_url}")
        except Exception as e:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Real vLLM backend unavailable for model '{model_name}' at {base_url}: {e}"
                ) from e
            logger.warning(f"Failed to initialize vLLM: {e}. Using fallback mode.")
            self.llm = None

        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Embedding model loaded: {embedding_model}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None

        # Market analysis prompt template
        self.analysis_prompt = PromptTemplate(
            input_variables=["market_data", "news", "technical_indicators"],
            template="""You are an expert financial analyst. Analyze the following market data and provide trading insights.

Market Data:
{market_data}

Recent News & Sentiment:
{news}

Technical Indicators:
{technical_indicators}

Provide a concise analysis with:
1. Market Trend (Bullish/Bearish/Neutral)
2. Key Risk Factors
3. Trading Signal Confidence (0-100)
4. Recommended Action (Buy/Sell/Hold)
5. Price Targets

Format your response as JSON.""",
        )

        # Sentiment analysis prompt
        self.sentiment_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Analyze the sentiment of the following market-related text. 
Return ONLY a JSON object with: {{"sentiment": "positive|negative|neutral", "score": 0-1, "confidence": 0-1}}

Text: {text}""",
        )

    def analyze_market_data(
        self,
        price_data: pd.DataFrame,
        technical_indicators: dict,
        news_items: list = None,
    ) -> dict:
        """
        Analyze market data using LLM.

        Args:
            price_data: DataFrame with OHLCV data
            technical_indicators: Dictionary of technical indicators
            news_items: List of recent news/market data

        Returns:
            Dictionary with analysis results
        """
        if self.llm is None:
            logger.warning("LLM not available, returning basic analysis")
            return self._fallback_analysis(price_data, technical_indicators)

        try:
            # Prepare market data summary
            market_summary = self._prepare_market_summary(price_data)

            # Process news sentiment
            news_summary = ""
            if news_items:
                news_summary = self._analyze_news_sentiment(news_items)

            # Prepare technical indicators summary
            indicators_text = self._format_indicators(technical_indicators)

            # Create chain and run analysis
            chain = self.analysis_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "market_data": market_summary,
                "news": news_summary or "No recent news available",
                "technical_indicators": indicators_text,
            })

            # Parse response
            analysis = self._parse_llm_response(response)
            analysis["timestamp"] = datetime.now().isoformat()
            analysis["model"] = self.model_name

            logger.info(f"Market analysis completed: {analysis['trend']}")
            return analysis

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(price_data, technical_indicators)

    def _analyze_news_sentiment(self, news_items: list) -> str:
        """Analyze sentiment from news items."""
        if not news_items or not self.llm:
            return ""

        sentiments = []
        for item in news_items[:5]:  # Analyze top 5 items
            text = item.get("text", item.get("title", ""))[:500]

            try:
                chain = self.sentiment_prompt | self.llm | StrOutputParser()
                response = chain.invoke({"text": text})
                sentiment_data = json.loads(response)
                sentiments.append(
                    {
                        "text": text[:100],
                        **sentiment_data,
                    }
                )
            except Exception as e:
                logger.debug(f"Sentiment analysis failed for item: {e}")

        if sentiments:
            avg_sentiment_score = np.mean([s["score"] for s in sentiments])
            return json.dumps(
                {
                    "items_analyzed": len(sentiments),
                    "average_sentiment": avg_sentiment_score,
                    "items": sentiments,
                }
            )
        return ""

    def _prepare_market_summary(self, price_data: pd.DataFrame) -> str:
        """Prepare market data summary for LLM."""
        if len(price_data) == 0:
            return "No price data available"

        latest = price_data.iloc[-1]
        previous = price_data.iloc[-2] if len(price_data) > 1 else latest

        price_change = ((latest["close"] - previous["close"]) / previous["close"]) * 100

        summary = f"""
Current Price: ${latest['close']:.2f}
High (24h): ${price_data['high'].max():.2f}
Low (24h): ${price_data['low'].min():.2f}
Volume: {latest.get('volume', 0):,.0f}
Price Change: {price_change:+.2f}%
Candles Analyzed: {len(price_data)}
"""
        return summary.strip()

    def _format_indicators(self, indicators: dict) -> str:
        """Format technical indicators for LLM."""
        formatted = []
        for key, value in indicators.items():
            if isinstance(value, (int, float)):
                formatted.append(f"{key}: {value:.2f}")
            else:
                formatted.append(f"{key}: {value}")
        return "\n".join(formatted)

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return {
                    "trend": analysis.get("trend", "NEUTRAL").upper(),
                    "confidence": analysis.get("confidence", 50),
                    "signal": analysis.get("signal", "HOLD").upper(),
                    "risk_factors": analysis.get("risk_factors", []),
                    "price_targets": analysis.get("price_targets", {}),
                    "raw_analysis": response,
                }
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")

        # Fallback parsing
        trend = "NEUTRAL"
        if "bullish" in response.lower():
            trend = "BULLISH"
        elif "bearish" in response.lower():
            trend = "BEARISH"

        return {
            "trend": trend,
            "confidence": 50,
            "signal": "HOLD",
            "raw_analysis": response,
        }

    def _fallback_analysis(self, price_data: pd.DataFrame, indicators: dict) -> dict:
        """Fallback analysis when LLM is unavailable."""
        logger.info("Using fallback analysis")

        # Simple trend detection
        if len(price_data) < 2:
            trend = "NEUTRAL"
        else:
            recent_change = (
                (price_data.iloc[-1]["close"] - price_data.iloc[-5]["close"])
                / price_data.iloc[-5]["close"]
            )
            if recent_change > 0.02:
                trend = "BULLISH"
            elif recent_change < -0.02:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

        return {
            "trend": trend,
            "confidence": 40,
            "signal": "HOLD",
            "model": "fallback",
            "timestamp": datetime.now().isoformat(),
        }

    def semantic_search(self, query: str, documents: list) -> list:
        """
        Perform semantic search on documents using embeddings.

        Args:
            query: Search query
            documents: List of documents to search

        Returns:
            List of top matching documents with scores
        """
        if not self.embedding_model or not documents:
            return []

        try:
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            doc_embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)

            cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
            top_results = np.argsort(-cos_scores.cpu())[:5]

            return [
                {
                    "text": documents[idx],
                    "score": float(cos_scores[idx]),
                }
                for idx in top_results
            ]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
