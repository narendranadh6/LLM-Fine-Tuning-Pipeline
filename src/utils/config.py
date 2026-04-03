"""
Configuration loader for the LoRA fine-tuning pipeline.

Loads and merges YAML config files (training + LoRA) into a unified
:class:`PipelineConfig` dataclass so that every module can access
strongly-typed settings without raw dict lookups.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str = "gpt2"
    revision: str = "main"
    trust_remote_code: bool = False
    torch_dtype: str = "float16"


@dataclass
class DatasetConfig:
    name: str = "wikitext"
    subset: str = "wikitext-103-raw-v1"
    train_split: str = "train"
    validation_split: str = "validation"
    text_column: str = "text"
    max_samples: Optional[int] = None


@dataclass
class TokenizerConfig:
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    padding_side: str = "right"


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "none"
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False


@dataclass
class EvaluationConfig:
    output_dir: str = "outputs/eval"
    toxicity_threshold: float = 0.5
    compute_bleu: bool = False
    compute_rouge: bool = True


@dataclass
class AWSConfig:
    s3_bucket: Optional[str] = None
    s3_prefix: str = "llm-lora"
    region: str = "us-east-1"
    upload_dataset: bool = False
    upload_model: bool = False


@dataclass
class ExperimentTrackingConfig:
    enabled: bool = False
    backend: str = "mlflow"
    experiment_name: str = "lora-finetuning"
    run_name: Optional[str] = None
    mlflow_tracking_uri: str = "mlruns"
    wandb_project: str = "lora-finetuning"


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    merge_weights: bool = False
    modules_to_save: Optional[List[str]] = None


@dataclass
class PipelineConfig:
    """Top-level config aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    experiment_tracking: ExperimentTrackingConfig = field(
        default_factory=ExperimentTrackingConfig
    )
    lora: LoRAConfig = field(default_factory=LoRAConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _from_dict(dataclass_type, data: Dict[str, Any]):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    valid_keys = {f.name for f in dataclass_type.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return dataclass_type(**filtered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    training_config_path: str | Path = "configs/training.yaml",
    lora_config_path: str | Path = "configs/lora.yaml",
) -> PipelineConfig:
    """Load and merge training + LoRA YAML configs into a :class:`PipelineConfig`.

    Args:
        training_config_path: Path to the training configuration YAML.
        lora_config_path:     Path to the LoRA configuration YAML.

    Returns:
        A fully populated :class:`PipelineConfig` instance.
    """
    training_dict = _load_yaml(training_config_path)
    lora_dict = _load_yaml(lora_config_path)

    # Merge LoRA section into training dict under "lora" key.
    merged = copy.deepcopy(training_dict)
    merged["lora"] = lora_dict.get("lora", {})

    cfg = PipelineConfig(
        model=_from_dict(ModelConfig, merged.get("model", {})),
        dataset=_from_dict(DatasetConfig, merged.get("dataset", {})),
        tokenizer=_from_dict(TokenizerConfig, merged.get("tokenizer", {})),
        training=_from_dict(TrainingConfig, merged.get("training", {})),
        evaluation=_from_dict(EvaluationConfig, merged.get("evaluation", {})),
        aws=_from_dict(AWSConfig, merged.get("aws", {})),
        experiment_tracking=_from_dict(
            ExperimentTrackingConfig, merged.get("experiment_tracking", {})
        ),
        lora=_from_dict(LoRAConfig, merged.get("lora", {})),
    )
    return cfg
