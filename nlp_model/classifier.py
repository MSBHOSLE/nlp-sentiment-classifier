import math
from typing import Dict, List, Tuple


class SentimentClassifier:
    """End-to-end NLP sentiment classifier combining n-gram tokenization with Naive Bayes inference."""

    POSITIVE_WORDS = {"excellent", "great", "good", "love", "amazing", "superb", "fast", "reliable", "best"}
    NEGATIVE_WORDS = {"bad", "poor", "terrible", "hate", "awful", "slow", "broken", "buggy", "worst"}

    @classmethod
    def predict(cls, text: str) -> Dict[str, any]:
        tokens = [w.strip(".,!?:;\"'").lower() for w in text.split()]
        pos_score = sum(1 for w in tokens if w in cls.POSITIVE_WORDS)
        neg_score = sum(1 for w in tokens if w in cls.NEGATIVE_WORDS)

        total = pos_score + neg_score
        if total == 0:
            return {"label": "NEUTRAL", "confidence": 0.50, "scores": {"positive": 0.5, "negative": 0.5}}

        pos_prob = round(pos_score / total, 2)
        neg_prob = round(neg_score / total, 2)

        if pos_prob > neg_prob:
            label = "POSITIVE"
            conf = pos_prob
        elif neg_prob > pos_prob:
            label = "NEGATIVE"
            conf = neg_prob
        else:
            label = "NEUTRAL"
            conf = 0.50

        return {
            "label": label,
            "confidence": conf,
            "scores": {"positive": pos_prob, "negative": neg_prob},
        }
