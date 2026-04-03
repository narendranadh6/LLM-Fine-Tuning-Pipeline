"""
Training entry-point for the LoRA fine-tuning pipeline.

Usage:
    python src/training/train.py --config configs/training.yaml \
                                  --lora_config configs/lora.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling

from src.data.preprocess import load_and_preprocess, load_cached_dataset
from src.data.tokenizer import load_tokenizer, tokenize_dataset
from src.training.lora_config import build_lora_config
from src.training.trainer import LoRATrainer, upload_to_s3
from src.utils.config import PipelineConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="outputs/logs/train.log")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_base_model(cfg: PipelineConfig) -> AutoModelForCausalLM:
    """Load the base causal LM and configure it for LoRA fine-tuning.

    Args:
        cfg: Full pipeline config.

    Returns:
        The base model with gradient checkpointing enabled (if configured).
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.model.torch_dtype, torch.float16)

    logger.info("Loading base model '%s' (dtype=%s).", cfg.model.name, cfg.model.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required for gradient checkpointing with PEFT.
        model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled.")

    return model


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------


def build_training_args(cfg: PipelineConfig) -> TrainingArguments:
    """Translate :class:`TrainingConfig` into HuggingFace :class:`TrainingArguments`.

    Args:
        cfg: Full pipeline config.

    Returns:
        A fully configured :class:`TrainingArguments` instance.
    """
    t = cfg.training
    output_dir = Path(t.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        fp16=t.fp16 and torch.cuda.is_available(),
        bf16=t.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=t.gradient_checkpointing,
        logging_steps=t.logging_steps,
        evaluation_strategy="steps",
        eval_steps=t.eval_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        report_to=t.report_to if t.report_to != "none" else "none",
        seed=t.seed,
        dataloader_num_workers=t.dataloader_num_workers,
        remove_unused_columns=t.remove_unused_columns,
    )


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(
    training_config_path: str = "configs/training.yaml",
    lora_config_path: str = "configs/lora.yaml",
    processed_data_path: str | None = None,
) -> None:
    """Full LoRA fine-tuning pipeline.

    Steps:
    1. Load config.
    2. Preprocess / load dataset.
    3. Tokenise dataset.
    4. Load base model and apply LoRA adapters.
    5. Train.
    6. Save final LoRA checkpoint.
    7. (Optionally) upload to S3.

    Args:
        training_config_path: Path to the training YAML config.
        lora_config_path:     Path to the LoRA YAML config.
        processed_data_path:  If provided, skip preprocessing and load from disk.
    """
    cfg = load_config(training_config_path, lora_config_path)
    logger.info("Pipeline config loaded.")

    # ---- Dataset ----
    if processed_data_path and Path(processed_data_path).exists():
        logger.info("Loading pre-processed dataset from '%s'.", processed_data_path)
        dataset = load_cached_dataset(processed_data_path)
    else:
        save_path = Path("data/processed")
        dataset = load_and_preprocess(cfg.dataset, save_path=save_path)

    # ---- Tokeniser ----
    tokenizer = load_tokenizer(cfg.model.name, cfg.tokenizer)
    tokenised_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        cfg.tokenizer,
        text_column=cfg.dataset.text_column,
    )

    # ---- Model ----
    base_model = load_base_model(cfg)
    lora_config = build_lora_config(cfg.lora)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ---- Training args ----
    training_args = build_training_args(cfg)

    # ---- Data collator (dynamic padding for efficiency) ----
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # CLM — not masked LM
    )

    # ---- Trainer ----
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset["train"],
        eval_dataset=tokenised_dataset.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        aws_cfg=cfg.aws,
        tracking_cfg=cfg.experiment_tracking,
    )

    # ---- Train ----
    logger.info("Starting training...")
    trainer.train()

    # ---- Save ----
    final_output = Path(cfg.training.output_dir) / "final"
    logger.info("Saving final LoRA checkpoint to '%s'.", final_output)
    model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))

    # ---- Upload to S3 ----
    if cfg.aws.upload_model:
        logger.info("Uploading model artifacts to S3...")
        upload_to_s3(str(final_output), cfg.aws)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        help="Path to the training YAML config file.",
    )
    parser.add_argument(
        "--lora_config",
        default="configs/lora.yaml",
        help="Path to the LoRA YAML config file.",
    )
    parser.add_argument(
        "--processed_data",
        default=None,
        help="Path to a pre-processed dataset (skips preprocessing step).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        training_config_path=args.config,
        lora_config_path=args.lora_config,
        processed_data_path=args.processed_data,
    )
