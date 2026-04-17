"""
News collection and sentiment analysis.
Uses GNews API for headlines, SimHash for deduplication.
"""

import hashlib
import os
import logging
from datetime import datetime, timedelta

import numpy as np
import requests

from config import DataConfig

logger = logging.getLogger(__name__)


def _simhash_64(text: str) -> int:
    """Compute a 64-bit SimHash for near-duplicate detection."""
    tokens = text.lower().split()
    v = [0] * 64
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(64):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class NewsSentimentCollector:
    """Collects crypto news and extracts sentiment vectors."""

    def __init__(self, config: DataConfig):
        self.api_key = config.gnews_api_key or os.getenv("GNEWS_API_KEY", "")
        self.base_url = "https://gnews.io/api/v4"
        self._seen_hashes: list[int] = []
        self._sentiment_model = None

    def fetch_news(self, query: str = "cryptocurrency bitcoin ethereum solana", max_articles: int = 20) -> list[dict]:
        """Fetch recent crypto news articles from GNews."""
        if not self.api_key:
            logger.info("No GNews API key — returning empty news list")
            return []

        params = {
            "q": query,
            "lang": "en",
            "max": max_articles,
            "sortby": "publishedAt",
            "token": self.api_key,
        }
        try:
            resp = requests.get(f"{self.base_url}/search", params=params, timeout=15)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
        except Exception as e:
            logger.warning("GNews fetch failed: %s", e)
            return []

        # Deduplicate via SimHash
        deduplicated = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            h = _simhash_64(text)
            is_dup = any(_hamming_distance(h, prev) < 4 for prev in self._seen_hashes)
            if not is_dup:
                self._seen_hashes.append(h)
                deduplicated.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "published_at": article.get("publishedAt", ""),
                    "url": article.get("url", ""),
                })
        # Keep hash list bounded
        self._seen_hashes = self._seen_hashes[-500:]
        return deduplicated

    def compute_sentiment_vector(self, articles: list[dict]) -> np.ndarray:
        """
        Extract a sentiment vector from news articles.
        Uses a simple keyword-based approach as fallback;
        replace with a proper sentiment model for production.
        """
        if not articles:
            return np.zeros(5)  # neutral

        # Keyword-based sentiment (production: use a frozen sentiment-LM)
        positive_words = {
            "bullish", "surge", "rally", "breakout", "gains", "soar", "pump",
            "adoption", "institutional", "upgrade", "partnership", "approval",
            "record", "high", "growth", "momentum", "accumulation",
        }
        negative_words = {
            "bearish", "crash", "dump", "plunge", "selloff", "fear", "hack",
            "exploit", "regulation", "ban", "lawsuit", "sec", "fraud",
            "liquidation", "decline", "drop", "loss", "risk",
        }
        asset_keywords = {
            "btc": {"bitcoin", "btc"},
            "eth": {"ethereum", "eth", "ether"},
            "sol": {"solana", "sol"},
        }

        scores = []
        for article in articles:
            text = f"{article['title']} {article['description']}".lower()
            words = set(text.split())
            pos = len(words & positive_words)
            neg = len(words & negative_words)
            total = pos + neg
            score = (pos - neg) / max(total, 1)
            scores.append(score)

        # Build 5-dim vector: [overall_sentiment, btc_sentiment, eth_sentiment, sol_sentiment, volume_signal]
        overall = np.mean(scores) if scores else 0.0

        asset_sentiments = {}
        for asset, kws in asset_keywords.items():
            asset_scores = []
            for article, score in zip(articles, scores):
                text = f"{article['title']} {article['description']}".lower()
                if any(kw in text for kw in kws):
                    asset_scores.append(score)
            asset_sentiments[asset] = np.mean(asset_scores) if asset_scores else 0.0

        volume_signal = min(len(articles) / 20.0, 1.0)  # news volume normalized

        return np.array([
            np.clip(overall, -1, 1),
            np.clip(asset_sentiments.get("btc", 0), -1, 1),
            np.clip(asset_sentiments.get("eth", 0), -1, 1),
            np.clip(asset_sentiments.get("sol", 0), -1, 1),
            volume_signal,
        ])

    def get_news_digest(self) -> dict:
        """Get a complete news digest with articles and sentiment."""
        articles = self.fetch_news()
        sentiment = self.compute_sentiment_vector(articles)

        # Build text summary for LLM input
        summaries = []
        for a in articles[:10]:  # top 10 for context window
            summaries.append(f"[{a['source']}] {a['title']}: {a['description']}")

        return {
            "articles": articles,
            "sentiment_vector": sentiment.tolist(),
            "news_summary": "\n".join(summaries) if summaries else "No recent crypto news available.",
            "article_count": len(articles),
        }
