"""
Evaluation metrics module.

Computes:
- Perplexity from model cross-entropy loss.
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
- BLEU score (optional, for seq2seq tasks).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_perplexity(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Estimate perplexity on a held-out dataset.

    Perplexity = exp(mean cross-entropy loss over all tokens).

    Args:
        model:      Causal LM model in evaluation mode.
        dataloader: DataLoader yielding batches with ``input_ids``, ``labels``,
                    and ``attention_mask``.
        device:     Torch device to run inference on.

    Returns:
        Perplexity score (lower = better).
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(dataloader, desc="Computing perplexity"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        if not torch.isnan(loss) and not torch.isinf(loss):
            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        logger.warning("No valid batches found; returning inf perplexity.")
        return float("inf")

    avg_loss = total_loss / total_batches
    perplexity = math.exp(min(avg_loss, 512))
    logger.info("Perplexity: %.4f (avg_loss=%.4f)", perplexity, avg_loss)
    return perplexity


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        predictions: List of generated / predicted strings.
        references:  List of reference (ground-truth) strings.

    Returns:
        Dict with keys ``rouge1``, ``rouge2``, ``rougeL`` (F-scores).
    """
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        logger.error("rouge_score not installed. Run: pip install rouge-score")
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    totals: Dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    count = 0

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
        count += 1

    if count == 0:
        return totals

    averaged = {k: v / count for k, v in totals.items()}
    logger.info("ROUGE scores: %s", averaged)
    return averaged


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------


def compute_bleu(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute corpus-level BLEU score.

    Args:
        predictions: List of hypothesis strings.
        references:  List of reference strings.

    Returns:
        BLEU score in [0, 1].
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # type: ignore
        import nltk  # type: ignore

        nltk.download("punkt", quiet=True)
    except ImportError:
        logger.error("nltk not installed. Run: pip install nltk")
        return 0.0

    hypothesis = [pred.split() for pred in predictions]
    reference_list = [[ref.split()] for ref in references]

    smoother = SmoothingFunction().method1
    score = corpus_bleu(reference_list, hypothesis, smoothing_function=smoother)
    logger.info("BLEU score: %.4f", score)
    return score
