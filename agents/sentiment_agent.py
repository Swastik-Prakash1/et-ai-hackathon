"""sentiment_agent.py
=================================
Financial sentiment agent powered by FinBERT.

This module produces the exact schema required by CONTEXT.md:
{
    "ticker": str,
    "sentiment_score": float,          # -1 to +1
    "sentiment_label": str,            # Positive / Negative / Neutral
    "tone_shift": bool,
    "tone_shift_direction": str,       # positive / negative
    "key_sentences": [
        {"text": str, "label": str, "score": float}
    ],
    "document_type": str
}
"""

import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
DEFAULT_DOCUMENT_TYPE = "earnings_call"
SENTENCE_MIN_LEN = 20
TONE_SHIFT_THRESHOLD = 0.20
KEY_SENTENCE_LIMIT = 5

_POSITIVE_HINTS = {
    "beat",
    "beats",
    "growth",
    "strong",
    "improve",
    "improved",
    "raised",
    "increase",
    "upside",
    "record",
    "profit",
    "margin expansion",
}
_NEGATIVE_HINTS = {
    "miss",
    "decline",
    "weak",
    "downgrade",
    "drop",
    "pressure",
    "fall",
    "loss",
    "headwind",
    "cut guidance",
    "lower",
}

logger = logging.getLogger(__name__)


def _load_finbert_pipeline() -> Optional[Any]:
    """Load FinBERT once at module import and return the pipeline or None on failure."""
    try:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer, pipeline

        model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME, num_labels=3)
        tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        device = 0 if torch.cuda.is_available() else -1

        logger.info("Loaded FinBERT model '%s' on device=%s", FINBERT_MODEL_NAME, device)
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        logger.warning("FinBERT load failed, using rule-based fallback: %s", exc)
        return None


SENTIMENT_PIPELINE = _load_finbert_pipeline()


def _split_sentences(text: str) -> list[str]:
    """Split financial text into usable sentences and drop obvious boilerplate."""
    if not text:
        return []

    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    cleaned: list[str] = []
    for sentence in raw_sentences:
        s = sentence.strip()
        if len(s) < SENTENCE_MIN_LEN:
            continue
        lowered = s.lower()
        if "safe harbor" in lowered or "forward-looking statement" in lowered:
            continue
        cleaned.append(s)
    return cleaned


def _label_to_signed_score(label: str, confidence: float) -> float:
    """Convert class label and confidence into a signed score in [-1, 1]."""
    if label == "Positive":
        return float(confidence)
    if label == "Negative":
        return float(-confidence)
    return 0.0


def _rule_based_sentence_sentiment(sentence: str) -> dict[str, Any]:
    """Provide a deterministic fallback sentiment when FinBERT is unavailable."""
    lowered = sentence.lower()
    pos_hits = sum(1 for hint in _POSITIVE_HINTS if hint in lowered)
    neg_hits = sum(1 for hint in _NEGATIVE_HINTS if hint in lowered)

    if pos_hits > neg_hits:
        return {"label": "Positive", "score": 0.60}
    if neg_hits > pos_hits:
        return {"label": "Negative", "score": 0.60}
    return {"label": "Neutral", "score": 0.55}


def _predict_sentence_sentiments(sentences: list[str]) -> list[dict[str, Any]]:
    """Predict sentence-level sentiment using FinBERT, with safe fallback per sentence."""
    results: list[dict[str, Any]] = []
    if not sentences:
        return results

    if SENTIMENT_PIPELINE is None:
        for sentence in sentences:
            pred = _rule_based_sentence_sentiment(sentence)
            results.append({"text": sentence, "label": pred["label"], "score": float(pred["score"])})
        return results

    batch_size = 16
    for start_idx in range(0, len(sentences), batch_size):
        batch = sentences[start_idx : start_idx + batch_size]
        try:
            preds = SENTIMENT_PIPELINE(batch)
            for sentence, pred in zip(batch, preds):
                label = str(pred.get("label", "Neutral"))
                score = float(pred.get("score", 0.0))
                if label not in {"Positive", "Negative", "Neutral"}:
                    label = "Neutral"
                results.append({"text": sentence, "label": label, "score": round(score, 4)})
        except Exception as exc:
            logger.warning("FinBERT inference failed for batch, using fallback: %s", exc)
            for sentence in batch:
                pred = _rule_based_sentence_sentiment(sentence)
                results.append({"text": sentence, "label": pred["label"], "score": float(pred["score"])})

    return results


def _compute_document_sentiment(predictions: list[dict[str, Any]]) -> tuple[float, str]:
    """Aggregate sentence predictions into sentiment_score and sentiment_label."""
    if not predictions:
        return 0.0, "Neutral"

    signed_scores = [_label_to_signed_score(p["label"], float(p["score"])) for p in predictions]
    sentiment_score = max(-1.0, min(1.0, float(sum(signed_scores) / len(signed_scores))))

    if sentiment_score > 0.10:
        sentiment_label = "Positive"
    elif sentiment_score < -0.10:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return round(sentiment_score, 4), sentiment_label


def _detect_tone_shift(current_score: float, previous_score: Optional[float]) -> tuple[bool, str]:
    """Detect whether tone shifted quarter-over-quarter and return (shifted, direction)."""
    if previous_score is None:
        return False, "positive"

    delta = float(current_score) - float(previous_score)
    tone_shift = abs(delta) >= TONE_SHIFT_THRESHOLD
    tone_shift_direction = "positive" if delta >= 0 else "negative"
    return tone_shift, tone_shift_direction


def analyze_sentiment(
    ticker: str,
    text: str,
    document_type: str = DEFAULT_DOCUMENT_TYPE,
    previous_text: Optional[str] = None,
    previous_score: Optional[float] = None,
) -> dict[str, Any]:
    """Analyze a financial text document and return the fixed sentiment agent schema.

    Args:
        ticker: NSE ticker without .NS suffix (e.g., RELIANCE).
        text: Current quarter text (earnings call / management commentary).
        document_type: Source type label for downstream use.
        previous_text: Optional previous-quarter text for tone shift detection.
        previous_score: Optional previous-quarter score if already computed.

    Returns:
        Dict matching CONTEXT.md sentiment_agent output schema exactly.
    """
    normalized_ticker = ticker.strip().upper().replace(".NS", "")
    current_sentences = _split_sentences(text)
    current_predictions = _predict_sentence_sentiments(current_sentences)
    sentiment_score, sentiment_label = _compute_document_sentiment(current_predictions)

    key_sentences = sorted(
        current_predictions,
        key=lambda item: abs(_label_to_signed_score(item["label"], float(item["score"]))),
        reverse=True,
    )[:KEY_SENTENCE_LIMIT]
    key_sentences = [
        {
            "text": item["text"],
            "label": item["label"],
            "score": round(float(item["score"]), 4),
        }
        for item in key_sentences
    ]

    inferred_previous_score = previous_score
    if inferred_previous_score is None and previous_text:
        prev_predictions = _predict_sentence_sentiments(_split_sentences(previous_text))
        inferred_previous_score, _ = _compute_document_sentiment(prev_predictions)

    tone_shift, tone_shift_direction = _detect_tone_shift(sentiment_score, inferred_previous_score)

    return {
        "ticker": normalized_ticker,
        "sentiment_score": float(sentiment_score),
        "sentiment_label": sentiment_label,
        "tone_shift": bool(tone_shift),
        "tone_shift_direction": tone_shift_direction,
        "key_sentences": key_sentences,
        "document_type": document_type,
    }


def smoke_test() -> dict[str, Any]:
    """Run a local smoke test for the sentiment pipeline and return the result payload."""
    current_doc = (
        "Revenue guidance raised for FY27 with strong demand across domestic and export segments. "
        "Operating margins improved for the third quarter in a row and management expects continued growth. "
        "The company announced capacity expansion backed by healthy cash generation."
    )
    previous_doc = (
        "Demand remained mixed this quarter and cost pressures affected profitability. "
        "Management stayed cautious on near-term guidance due to macro headwinds."
    )

    return analyze_sentiment(
        ticker="RELIANCE",
        text=current_doc,
        document_type="earnings_call",
        previous_text=previous_doc,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    output = smoke_test()
    print(output)
