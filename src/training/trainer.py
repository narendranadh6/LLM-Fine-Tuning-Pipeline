"""
Custom HuggingFace Trainer subclass for LoRA fine-tuning.

Extends :class:`transformers.Trainer` to:
- Log perplexity alongside loss.
- Integrate optional experiment tracking (W&B / MLflow).
- Expose a utility for AWS S3 checkpoint upload.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerControl, TrainerState

from src.utils.config import AWSConfig, ExperimentTrackingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class PerplexityLoggingCallback(TrainerCallback):
    """Log validation perplexity at the end of every evaluation phase."""

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs: Any,
    ) -> None:
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            perplexity = math.exp(min(eval_loss, 512))  # cap to avoid overflow
            metrics["eval_perplexity"] = perplexity
            logger.info("Step %d | eval_loss=%.4f | eval_perplexity=%.2f",
                        state.global_step, eval_loss, perplexity)


class ExperimentTrackingCallback(TrainerCallback):
    """Route metrics to W&B or MLflow when enabled."""

    def __init__(self, cfg: ExperimentTrackingConfig) -> None:
        self.cfg = cfg
        self._client: Any = None

        if not cfg.enabled:
            return

        if cfg.backend == "wandb":
            try:
                import wandb  # noqa: PLC0415

                wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.run_name,
                    tags=["lora", "finetuning"],
                )
                self._client = wandb
                logger.info("W&B experiment tracking enabled (project=%s).", cfg.wandb_project)
            except ImportError:
                logger.warning("wandb not installed; skipping W&B tracking.")

        elif cfg.backend == "mlflow":
            try:
                import mlflow  # noqa: PLC0415

                mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
                mlflow.set_experiment(cfg.experiment_name)
                mlflow.start_run(run_name=cfg.run_name)
                self._client = mlflow
                logger.info("MLflow experiment tracking enabled (experiment=%s).", cfg.experiment_name)
            except ImportError:
                logger.warning("mlflow not installed; skipping MLflow tracking.")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if not self.cfg.enabled or self._client is None:
            return

        step = state.global_step
        if self.cfg.backend == "wandb":
            self._client.log(logs, step=step)
        elif self.cfg.backend == "mlflow":
            self._client.log_metrics({k: v for k, v in logs.items() if isinstance(v, (int, float))}, step=step)

    def on_train_end(self, *args: Any, **kwargs: Any) -> None:
        if not self.cfg.enabled or self._client is None:
            return
        if self.cfg.backend == "mlflow":
            self._client.end_run()


# ---------------------------------------------------------------------------
# S3 upload helper
# ---------------------------------------------------------------------------


def upload_to_s3(local_path: str, aws_cfg: AWSConfig, prefix_override: Optional[str] = None) -> None:
    """Upload a local file or directory to an S3 bucket.

    Args:
        local_path:      Local filesystem path to upload.
        aws_cfg:         AWS settings from the pipeline config.
        prefix_override: Override the S3 key prefix from cfg.
    """
    if aws_cfg.s3_bucket is None:
        logger.warning("S3 upload requested but no bucket configured; skipping.")
        return

    try:
        import boto3  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        s3 = boto3.client("s3", region_name=aws_cfg.region)
        prefix = prefix_override or aws_cfg.s3_prefix
        local = Path(local_path)

        if local.is_file():
            key = f"{prefix}/{local.name}"
            s3.upload_file(str(local), aws_cfg.s3_bucket, key)
            logger.info("Uploaded %s → s3://%s/%s", local, aws_cfg.s3_bucket, key)
        elif local.is_dir():
            for child in local.rglob("*"):
                if child.is_file():
                    rel = child.relative_to(local)
                    key = f"{prefix}/{rel}"
                    s3.upload_file(str(child), aws_cfg.s3_bucket, key)
            logger.info("Uploaded directory %s → s3://%s/%s/", local, aws_cfg.s3_bucket, prefix)
    except ImportError:
        logger.error("boto3 is not installed; cannot upload to S3.")
    except Exception as exc:  # noqa: BLE001
        logger.error("S3 upload failed: %s", exc)


# ---------------------------------------------------------------------------
# Custom Trainer
# ---------------------------------------------------------------------------


class LoRATrainer(Trainer):
    """HuggingFace Trainer extended with perplexity logging and S3 support."""

    def __init__(
        self,
        *args: Any,
        aws_cfg: Optional[AWSConfig] = None,
        tracking_cfg: Optional[ExperimentTrackingConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.aws_cfg = aws_cfg

        # Register custom callbacks.
        self.add_callback(PerplexityLoggingCallback())
        if tracking_cfg is not None:
            self.add_callback(ExperimentTrackingCallback(tracking_cfg))

    def log(self, logs: Dict[str, float]) -> None:  # type: ignore[override]
        """Override to add perplexity to training logs."""
        if "loss" in logs:
            try:
                logs["perplexity"] = math.exp(min(logs["loss"], 512))
            except OverflowError:
                logs["perplexity"] = float("inf")
        super().log(logs)

    def save_model_to_s3(self, output_dir: str) -> None:
        """Convenience wrapper to upload a checkpoint directory to S3."""
        if self.aws_cfg is not None:
            upload_to_s3(output_dir, self.aws_cfg, prefix_override=f"{self.aws_cfg.s3_prefix}/checkpoints")
