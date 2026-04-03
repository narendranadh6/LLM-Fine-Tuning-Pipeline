"""
Data preprocessing module for the LoRA fine-tuning pipeline.

Responsibilities:
- Load a HuggingFace dataset (or local files).
- Clean raw text (remove null/empty rows, normalise whitespace).
- Optionally subsample the dataset for quick experiments.
- Save the processed dataset to disk for downstream tokenisation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from src.utils.config import DatasetConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Normalise and clean a single text string.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Collapse multiple consecutive whitespace characters into a single space.
    3. Remove non-printable / null characters.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", "", text)
    return text


def _clean_batch(batch: dict, text_column: str) -> dict:
    """Apply :func:`_clean_text` to every row in a batch (for ``Dataset.map``)."""
    batch[text_column] = [_clean_text(t) for t in batch[text_column]]
    return batch


def _filter_empty(example: dict, text_column: str) -> bool:
    """Return ``True`` if the example's text is non-empty after cleaning."""
    return bool(example[text_column].strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_preprocess(
    cfg: DatasetConfig,
    save_path: Optional[str | Path] = None,
) -> DatasetDict:
    """Load a dataset, clean it, and optionally persist it.

    Args:
        cfg:       :class:`DatasetConfig` with dataset name / split info.
        save_path: If provided, the cleaned dataset is saved here using
                   ``DatasetDict.save_to_disk``.

    Returns:
        A :class:`DatasetDict` containing ``train`` and ``validation`` splits.
    """
    logger.info("Loading dataset '%s' (subset: %s)", cfg.name, cfg.subset or "default")

    # --- Load ---
    load_kwargs = {"path": cfg.name}
    if cfg.subset:
        load_kwargs["name"] = cfg.subset

    raw: DatasetDict = load_dataset(**load_kwargs)  # type: ignore[arg-type]

    splits_to_use = {}
    for split_key, split_name in [
        ("train", cfg.train_split),
        ("validation", cfg.validation_split),
    ]:
        if split_name in raw:
            splits_to_use[split_key] = raw[split_name]
        else:
            logger.warning("Split '%s' not found in dataset; skipping.", split_name)

    if not splits_to_use:
        raise ValueError(
            f"None of the requested splits ({cfg.train_split!r}, "
            f"{cfg.validation_split!r}) were found in dataset '{cfg.name}'."
        )

    dataset = DatasetDict(splits_to_use)

    # --- Subsample ---
    if cfg.max_samples is not None:
        logger.info("Subsampling each split to %d examples.", cfg.max_samples)
        dataset = DatasetDict(
            {
                split: ds.select(range(min(cfg.max_samples, len(ds))))
                for split, ds in dataset.items()
            }
        )

    # --- Clean ---
    logger.info("Cleaning text column '%s'.", cfg.text_column)
    dataset = dataset.map(
        _clean_batch,
        fn_kwargs={"text_column": cfg.text_column},
        batched=True,
        desc="Cleaning text",
    )
    dataset = dataset.filter(
        _filter_empty,
        fn_kwargs={"text_column": cfg.text_column},
        desc="Filtering empty rows",
    )

    for split, ds in dataset.items():
        logger.info("Split '%s': %d examples after cleaning.", split, len(ds))

    # --- Save ---
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info("Saving processed dataset to '%s'.", save_path)
        dataset.save_to_disk(str(save_path))

    return dataset


def load_cached_dataset(path: str | Path) -> DatasetDict:
    """Load a previously preprocessed dataset saved with :func:`load_and_preprocess`.

    Args:
        path: Directory created by ``DatasetDict.save_to_disk``.

    Returns:
        Restored :class:`DatasetDict`.
    """
    logger.info("Loading cached dataset from '%s'.", path)
    return load_from_disk(str(path))
