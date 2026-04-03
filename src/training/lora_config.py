"""
LoRA configuration builder for the fine-tuning pipeline.

Wraps PEFT's ``LoraConfig`` with sensible defaults and documents every
parameter so that configurations are reproducible and auditable.
"""

from __future__ import annotations

from peft import LoraConfig, TaskType

from src.utils.config import LoRAConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping from string task-type names to PEFT TaskType enum members.
_TASK_TYPE_MAP: dict[str, TaskType] = {
    "CAUSAL_LM": TaskType.CAUSAL_LM,
    "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    "TOKEN_CLS": TaskType.TOKEN_CLS,
    "SEQ_CLS": TaskType.SEQ_CLS,
}


def build_lora_config(cfg: LoRAConfig) -> LoraConfig:
    """Construct a PEFT :class:`LoraConfig` from a pipeline :class:`LoRAConfig`.

    Args:
        cfg: LoRA settings loaded from ``configs/lora.yaml``.

    Returns:
        A fully configured PEFT :class:`LoraConfig` ready to be passed to
        :func:`peft.get_peft_model`.
    """
    task_type = _TASK_TYPE_MAP.get(cfg.task_type.upper())
    if task_type is None:
        raise ValueError(
            f"Unknown task_type '{cfg.task_type}'. "
            f"Valid options: {list(_TASK_TYPE_MAP.keys())}"
        )

    logger.info(
        "Building LoraConfig | r=%d | alpha=%d | dropout=%.2f | "
        "target_modules=%s | task_type=%s",
        cfg.r,
        cfg.lora_alpha,
        cfg.lora_dropout,
        cfg.target_modules,
        task_type.value,
    )

    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        task_type=task_type,
        modules_to_save=cfg.modules_to_save,
    )

    # Log the derived effective scaling factor.
    effective_alpha = cfg.lora_alpha / cfg.r
    logger.info("Effective LoRA scaling factor (alpha/r) = %.2f", effective_alpha)

    return lora_config
