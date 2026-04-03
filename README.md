# Production-Grade LLM LoRA Fine-Tuning Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2%20%7C%20SageMaker-FF9900)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)](https://www.docker.com/)

A scalable, modular, and production-ready pipeline for fine-tuning Large Language Models using **LoRA (Low-Rank Adaptation)** via PEFT. Includes data preprocessing, training with gradient optimisations, multi-metric evaluation, bias/toxicity detection, AWS integration, and Docker support.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      Pipeline Architecture                         │
├──────────────┬─────────────────┬─────────────────┬────────────────┤
│  Data Layer  │ Training Layer  │ Evaluation Layer│ Inference Layer│
│              │                 │                 │                │
│ preprocess.py│  lora_config.py │  metrics.py     │  generate.py   │
│ tokenizer.py │  trainer.py     │  bias_toxicity  │                │
│              │  train.py       │  evaluate.py    │                │
├──────────────┴─────────────────┴─────────────────┴────────────────┤
│              Shared Utilities: config.py | logger.py               │
├───────────────────────────────────────────────────────────────────┤
│              Infrastructure: Dockerfile | AWS S3/EC2 | W&B/MLflow  │
└───────────────────────────────────────────────────────────────────┘
```

```
llm-lora-finetuning/
├── configs/
│   ├── training.yaml       # All training hyperparameters
│   └── lora.yaml           # LoRA rank, alpha, target modules
├── src/
│   ├── data/
│   │   ├── preprocess.py   # Dataset loading + cleaning
│   │   └── tokenizer.py    # Tokenisation with causal-LM labels
│   ├── training/
│   │   ├── lora_config.py  # PEFT LoraConfig builder
│   │   ├── trainer.py      # Custom HF Trainer + perplexity logging
│   │   └── train.py        # Training entry-point
│   ├── evaluation/
│   │   ├── metrics.py      # Perplexity, ROUGE, BLEU
│   │   ├── bias_toxicity.py# Toxicity & bias detectors
│   │   └── evaluate.py     # Evaluation entry-point
│   ├── inference/
│   │   └── generate.py     # Inference + base vs. LoRA comparison
│   └── utils/
│       ├── config.py       # YAML → typed dataclasses
│       └── logger.py       # Structured logging
├── scripts/
│   ├── run_training.sh
│   └── run_eval.sh
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/Production-Grade-LLM-LoRA-Finetuning.git
cd Production-Grade-LLM-LoRA-Finetuning

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Edit `configs/training.yaml` and `configs/lora.yaml` to set:
- `model.name` — base model (default: `gpt2`; swap to `meta-llama/Llama-2-7b-hf` etc.)
- `dataset.name` — HuggingFace dataset or path
- `lora.r`, `lora.lora_alpha`, `lora.target_modules` — LoRA hyperparameters
- `training.fp16`, `training.gradient_accumulation_steps` — GPU efficiency
- `aws.s3_bucket` — S3 bucket for artifact storage (optional)

### 3. Train

```bash
# Using Python directly
python src/training/train.py --config configs/training.yaml --lora_config configs/lora.yaml

# Using the shell script (adds logging & sanity checks)
bash scripts/run_training.sh
```

### 4. Evaluate

```bash
python src/evaluation/evaluate.py \
    --model_path outputs/checkpoints/final \
    --base_model gpt2

# Or via script
bash scripts/run_eval.sh --model_path outputs/checkpoints/final
```

### 5. Run Inference

```bash
# Generate with LoRA model
python src/inference/generate.py \
    --model_path outputs/checkpoints/final \
    --base_model gpt2 \
    --prompt "The future of artificial intelligence is"

# Side-by-side comparison: base vs. LoRA
python src/inference/generate.py \
    --model_path outputs/checkpoints/final \
    --base_model gpt2 \
    --prompt "Machine learning enables" \
    --compare
```

---

## LoRA Configuration

| Parameter | Default | Description |
|---|---|---|
| `r` | 16 | Rank of update matrices |
| `lora_alpha` | 32 | Scaling factor (effective scale = alpha/r = 2.0) |
| `lora_dropout` | 0.05 | Dropout for regularisation |
| `target_modules` | `c_attn, c_proj` | Modules to adapt (GPT-2) |
| `bias` | `none` | Which biases to train |

**For LLaMA / Mistral**, change `target_modules` to `q_proj, v_proj, k_proj, o_proj`.

---

## Training Optimisations

| Technique | Config Key | Effect |
|---|---|---|
| Gradient Checkpointing | `gradient_checkpointing: true` | ~60% VRAM reduction |
| Mixed Precision (fp16) | `fp16: true` | ~2× throughput on Tensor cores |
| Gradient Accumulation | `gradient_accumulation_steps: 8` | Simulates larger batch sizes |
| Efficient Batching | `per_device_train_batch_size` | Tuned per GPU VRAM |

Expected improvement: **~25–30% less training time** vs. full fine-tuning without LoRA.

---

## Expected Results

| Metric | Base Model | LoRA Fine-Tuned | Improvement |
|---|---|---|---|
| Perplexity | ~35–45 | ~27–36 | **~15–20% reduction** |
| ROUGE-1 | — | Measured vs. base | Domain-specific ↑ |
| Toxicity Rate | Baseline | Comparable / lower | Safe outputs |

*(Actual numbers depend on dataset, model size, and hardware.)*

---

## Experiment Tracking

### Weights & Biases
```yaml
# configs/training.yaml
experiment_tracking:
  enabled: true
  backend: "wandb"
  wandb_project: "lora-finetuning"
training:
  report_to: "wandb"
```

### MLflow
```yaml
experiment_tracking:
  enabled: true
  backend: "mlflow"
  mlflow_tracking_uri: "mlruns"
```

---

## Docker

### Build
```bash
docker build -t llm-lora-finetuning .
```

### Train
```bash
docker run --gpus all -v $(pwd)/outputs:/app/outputs llm-lora-finetuning \
    python src/training/train.py --config configs/training.yaml
```

### Evaluate
```bash
docker run --gpus all -v $(pwd)/outputs:/app/outputs llm-lora-finetuning \
    python src/evaluation/evaluate.py \
        --model_path outputs/checkpoints/final \
        --base_model gpt2
```

---

## AWS Deployment

### S3 Dataset & Model Storage

Configure in `configs/training.yaml`:
```yaml
aws:
  s3_bucket: "my-lora-training-bucket"
  s3_prefix: "llm-lora"
  region: "us-east-1"
  upload_dataset: true
  upload_model: true
```

Authenticate with AWS CLI:
```bash
aws configure  # or use IAM role on EC2/SageMaker
```

### EC2 Training

```bash
# 1. Launch a GPU instance (e.g., g4dn.xlarge with Deep Learning AMI)
# 2. SSH into instance and clone repo
git clone https://github.com/<username>/Production-Grade-LLM-LoRA-Finetuning.git
cd Production-Grade-LLM-LoRA-Finetuning
pip install -r requirements.txt

# 3. Run training
bash scripts/run_training.sh
```

### SageMaker

```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point="src/training/train.py",
    source_dir=".",
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    transformers_version="4.36",
    pytorch_version="2.1",
    py_version="py310",
    hyperparameters={
        "config": "configs/training.yaml",
        "lora_config": "configs/lora.yaml",
    },
)
estimator.fit({"train": s3_training_data_uri})
```

---

## Responsible AI Evaluation

The pipeline includes automated checks:

| Check | Tool / Method | Output |
|---|---|---|
| **Toxicity** | `unitary/toxic-bert` | Per-text scores + flagged indices |
| **Bias** | Zero-shot NLI (`bart-large-mnli`) | Bias category scores |

Results are saved to `outputs/eval/evaluation_results.json`.

---

## Reproducibility

Set `training.seed: 42` in `configs/training.yaml` for fully deterministic runs.
Pin exact package versions in `requirements.txt` before deploying to production.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'feat: add my feature'`)
4. Push and open a Pull Request
