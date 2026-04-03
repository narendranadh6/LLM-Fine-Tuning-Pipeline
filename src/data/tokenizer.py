"""
Tokenisation utilities for the LoRA fine-tuning pipeline.

Converts a raw-text :class:`DatasetDict` into model-ready token IDs using a
HuggingFace :class:`PreTrainedTokenizer`.  Supports causal LM (where labels ==
input_ids) and leaves room for seq2seq extension.
"""

from __future__ import annotations

from typing import Optional

from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.utils.config import TokenizerConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer factory
# ---------------------------------------------------------------------------


def load_tokenizer(model_name: str, cfg: TokenizerConfig) -> PreTrainedTokenizer:
    """Load a tokenizer from the HuggingFace hub and apply pipeline settings.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"gpt2"``).
        cfg:        :class:`TokenizerConfig` with padding / truncation settings.

    Returns:
        A configured :class:`PreTrainedTokenizer`.
    """
    logger.info("Loading tokenizer for model '%s'.", model_name)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )
    tokenizer.padding_side = cfg.padding_side

    # GPT-style models have no dedicated pad token — reuse EOS.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token ('%s').", tokenizer.eos_token)

    return tokenizer


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------


def _tokenize_batch(
    batch: dict,
    tokenizer: PreTrainedTokenizer,
    text_column: str,
    max_length: int,
    padding: str | bool,
    truncation: bool,
) -> dict:
    """Tokenise a batch and create ``labels`` equal to ``input_ids``.

    For causal language modelling the labels are identical to input_ids.
    The ``transformers`` Trainer automatically shifts them left internally.
    Padding tokens in labels are set to ``-100`` so cross-entropy ignores them.
    """
    tokenised = tokenizer(
        batch[text_column],
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=None,  # return plain Python lists for Arrow compatibility
    )

    # Build labels: copy input_ids but mask padding positions.
    pad_id = tokenizer.pad_token_id
    labels = []
    for ids, mask in zip(tokenised["input_ids"], tokenised["attention_mask"]):
        label_row = [
            token_id if attn == 1 else -100
            for token_id, attn in zip(ids, mask)
        ]
        labels.append(label_row)

    tokenised["labels"] = labels
    return tokenised


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    cfg: TokenizerConfig,
    text_column: str = "text",
    num_proc: Optional[int] = None,
    remove_raw_columns: bool = True,
) -> DatasetDict:
    """Apply tokenisation to every split of *dataset*.

    Args:
        dataset:            Input :class:`DatasetDict` (train / validation).
        tokenizer:          Configured tokenizer produced by :func:`load_tokenizer`.
        cfg:                :class:`TokenizerConfig` settings.
        text_column:        Name of the raw text column to tokenise.
        num_proc:           Number of worker processes for ``Dataset.map``.
        remove_raw_columns: Drop non-numeric columns after tokenisation.

    Returns:
        A tokenised :class:`DatasetDict` ready for the ``Trainer``.
    """
    logger.info(
        "Tokenising dataset (max_length=%d, padding=%s, truncation=%s).",
        cfg.max_length,
        cfg.padding,
        cfg.truncation,
    )

    # Columns to remove (keep only tensor-compatible fields).
    columns_to_remove = (
        [c for c in dataset["train"].column_names] if remove_raw_columns else []
    )

    tokenised = dataset.map(
        _tokenize_batch,
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_column": text_column,
            "max_length": cfg.max_length,
            "padding": cfg.padding,
            "truncation": cfg.truncation,
        },
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_remove,
        desc="Tokenising",
    )

    for split, ds in tokenised.items():
        logger.info("Split '%s': %d tokenised examples.", split, len(ds))

    return tokenised
