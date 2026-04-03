"""
Evaluation entry-point for the LoRA fine-tuning pipeline.

Usage:
    python src/evaluation/evaluate.py \
        --model_path outputs/checkpoints/final \
        --base_model gpt2 \
        --config configs/training.yaml \
        --lora_config configs/lora.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling

from src.data.preprocess import load_and_preprocess
from src.data.tokenizer import load_tokenizer, tokenize_dataset
from src.evaluation.bias_toxicity import detect_bias, detect_toxicity
from src.evaluation.metrics import compute_bleu, compute_perplexity, compute_rouge
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="outputs/logs/eval.log")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_lora_model(base_model_name: str, peft_checkpoint: str) -> AutoModelForCausalLM:
    """Load the base model and attach a LoRA adapter.

    Args:
        base_model_name: HuggingFace hub ID of the base model.
        peft_checkpoint: Path to the saved PEFT / LoRA adapter directory.

    Returns:
        A :class:`PeftModel` wrapping the base model.
    """
    logger.info("Loading base model '%s' for evaluation.", base_model_name)
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
    logger.info("Loading LoRA adapter from '%s'.", peft_checkpoint)
    model = PeftModel.from_pretrained(base, peft_checkpoint)
    model.eval()
    return model


def _load_base_only(base_model_name: str) -> AutoModelForCausalLM:
    """Load only the base model for baseline comparison."""
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Generation helper for RA checks
# ---------------------------------------------------------------------------


def _generate_samples(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """Generate text continuations for a list of prompts."""
    model.to(device)
    generated = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated.append(text)
    return generated


# ---------------------------------------------------------------------------
# Evaluation routine
# ---------------------------------------------------------------------------


def evaluate(
    model_path: str,
    base_model_name: str,
    training_config_path: str = "configs/training.yaml",
    lora_config_path: str = "configs/lora.yaml",
    processed_data_path: str | None = None,
) -> dict:
    """Run the full evaluation suite: perplexity, ROUGE, toxicity, bias.

    Args:
        model_path:             Path to the fine-tuned LoRA checkpoint.
        base_model_name:        HuggingFace hub ID of the base model.
        training_config_path:   Path to training YAML config.
        lora_config_path:       Path to LoRA YAML config.
        processed_data_path:    Optional pre-processed dataset path.

    Returns:
        Dict containing all metric results.
    """
    cfg = load_config(training_config_path, lora_config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation device: %s", device)

    # ---- Dataset ----
    if processed_data_path and Path(processed_data_path).exists():
        from src.data.preprocess import load_cached_dataset  # noqa: PLC0415 (lazy import)
        dataset = load_cached_dataset(processed_data_path)
    else:
        dataset = load_and_preprocess(cfg.dataset)

    tokenizer = load_tokenizer(base_model_name, cfg.tokenizer)
    tokenised = tokenize_dataset(dataset, tokenizer, cfg.tokenizer, text_column=cfg.dataset.text_column)

    val_ds = tokenised.get("validation")
    if val_ds is None:
        raise RuntimeError("No validation split found in the tokenised dataset.")

    val_ds.set_format("torch")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.per_device_eval_batch_size,
        collate_fn=collator,
    )

    results: dict = {}

    # ---- Perplexity comparison ----
    logger.info("--- Perplexity: Base Model ---")
    base_model = _load_base_only(base_model_name)
    base_ppl = compute_perplexity(base_model, val_loader, device)
    del base_model
    torch.cuda.empty_cache()

    logger.info("--- Perplexity: LoRA Fine-Tuned Model ---")
    lora_model = _load_lora_model(base_model_name, model_path)
    lora_ppl = compute_perplexity(lora_model, val_loader, device)

    ppl_improvement = (base_ppl - lora_ppl) / base_ppl * 100
    results["perplexity"] = {
        "base_model": base_ppl,
        "lora_model": lora_ppl,
        "improvement_pct": ppl_improvement,
    }
    logger.info(
        "Perplexity | base=%.2f | lora=%.2f | improvement=%.1f%%",
        base_ppl,
        lora_ppl,
        ppl_improvement,
    )

    # ---- ROUGE (on a small sample) ----
    if cfg.evaluation.compute_rouge:
        logger.info("--- Computing ROUGE ---")
        sample_texts = dataset["validation"][cfg.dataset.text_column][:50]
        # Use first 30 tokens as prompt, rest as reference.
        prompts = [" ".join(t.split()[:30]) for t in sample_texts]
        references = sample_texts
        predictions = _generate_samples(lora_model, tokenizer, prompts, device=device)
        rouge = compute_rouge(predictions, references)
        results["rouge"] = rouge

    # ---- BLEU (optional) ----
    if cfg.evaluation.compute_bleu:
        logger.info("--- Computing BLEU ---")
        bleu = compute_bleu(predictions, references)
        results["bleu"] = bleu

    # ---- Responsible AI ----
    logger.info("--- Responsible AI: Toxicity ---")
    sample_texts_rai = dataset["validation"][cfg.dataset.text_column][:100]
    generated_rai = _generate_samples(lora_model, tokenizer, sample_texts_rai[:20], device=device)
    toxicity_result = detect_toxicity(generated_rai, threshold=cfg.evaluation.toxicity_threshold)
    results["toxicity"] = toxicity_result

    logger.info("--- Responsible AI: Bias ---")
    bias_result = detect_bias(generated_rai[:10])
    results["bias"] = bias_result

    # ---- Save results ----
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        # Convert non-serialisable objects (e.g., floats from torch).
        json.dump(results, f, indent=2, default=str)
    logger.info("Evaluation results saved to '%s'.", results_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA fine-tuned model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the fine-tuned LoRA adapter directory.",
    )
    parser.add_argument(
        "--base_model",
        default="gpt2",
        help="HuggingFace hub ID of the base model.",
    )
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--lora_config", default="configs/lora.yaml")
    parser.add_argument("--processed_data", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = evaluate(
        model_path=args.model_path,
        base_model_name=args.base_model,
        training_config_path=args.config,
        lora_config_path=args.lora_config,
        processed_data_path=args.processed_data,
    )
    print(json.dumps(results, indent=2, default=str))
