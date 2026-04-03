"""
Inference module for LoRA fine-tuned models.

Loads a fine-tuned adapter, generates text for given prompts, and
optionally compares outputs side-by-side with the base model.

Usage:
    python src/inference/generate.py \
        --model_path outputs/checkpoints/final \
        --base_model gpt2 \
        --prompt "Machine learning is"
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def load_lora_model(
    base_model_name: str,
    peft_path: str,
    device: torch.device,
) -> tuple[PeftModel, AutoTokenizer]:
    """Load a LoRA fine-tuned model and its tokenizer.

    Args:
        base_model_name: HuggingFace hub ID of the base model.
        peft_path:       Path to the saved PEFT adapter checkpoint.
        device:          Target device for inference.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info("Loading LoRA model from '%s' (base: %s).", peft_path, base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(peft_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, peft_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_base_model(
    base_model_name: str,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load only the base model (no LoRA) for comparison.

    Args:
        base_model_name: HuggingFace hub ID.
        device:          Target device.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info("Loading base model '%s' for comparison.", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a text continuation for *prompt*.

    Args:
        model:              Causal LM (base or LoRA-adapted).
        tokenizer:          Corresponding tokenizer.
        prompt:             Input text to continue.
        device:             Inference device.
        max_new_tokens:     Maximum number of new tokens to generate.
        temperature:        Sampling temperature (lower = more deterministic).
        top_p:              Nucleus sampling probability mass.
        do_sample:          Enable nucleus/temperature sampling.
        repetition_penalty: Penalty for repeating tokens (> 1.0 discourages repeats).

    Returns:
        The generated text (including the original prompt).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=gen_config)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_outputs(
    base_model_name: str,
    peft_path: str,
    prompts: list[str],
    device: torch.device,
    **gen_kwargs,
) -> list[dict]:
    """Generate base vs. LoRA outputs for a list of prompts and return comparisons.

    Args:
        base_model_name: HuggingFace hub ID of the base model.
        peft_path:       LoRA adapter checkpoint path.
        prompts:         List of prompt strings.
        device:          Inference device.
        **gen_kwargs:    Additional keyword args forwarded to :func:`generate_text`.

    Returns:
        List of dicts with ``prompt``, ``base``, ``lora`` keys.
    """
    base_model, base_tok = load_base_model(base_model_name, device)
    lora_model, lora_tok = load_lora_model(base_model_name, peft_path, device)

    comparisons = []
    for prompt in prompts:
        base_out = generate_text(base_model, base_tok, prompt, device, **gen_kwargs)
        lora_out = generate_text(lora_model, lora_tok, prompt, device, **gen_kwargs)
        comparisons.append({"prompt": prompt, "base": base_out, "lora": lora_out})

        # Pretty print to console.
        separator = "=" * 60
        logger.info("\n%s\nPROMPT: %s\n\n[BASE MODEL]\n%s\n\n[LORA MODEL]\n%s\n%s",
                    separator, prompt, base_out, lora_out, separator)

    return comparisons


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with a LoRA fine-tuned model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="LoRA adapter checkpoint directory.")
    parser.add_argument("--base_model", default="gpt2", help="Base model name.")
    parser.add_argument("--prompt", default="The quick brown fox", help="Prompt text.")
    parser.add_argument("--compare", action="store_true", help="Compare with base model output.")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_sample", action="store_true", help="Use greedy decoding.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )

    if args.compare:
        compare_outputs(
            base_model_name=args.base_model,
            peft_path=args.model_path,
            prompts=[args.prompt],
            device=device,
            **gen_kwargs,
        )
    else:
        model, tokenizer = load_lora_model(args.base_model, args.model_path, device)
        output = generate_text(model, tokenizer, args.prompt, device, **gen_kwargs)
        print("\n" + "=" * 60)
        print(f"PROMPT: {args.prompt}")
        print("=" * 60)
        print(output)
