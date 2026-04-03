# ========================================================================
# Dockerfile — Production-grade LoRA fine-tuning pipeline
# ========================================================================
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ---- System dependencies ------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        git \
        wget \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Alias python3.10 → python & pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ---- Working directory --------------------------------------------------
WORKDIR /app

# ---- Python dependencies ------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Project source ------------------------------------------------------
COPY . .

# Ensure scripts are executable
RUN chmod +x scripts/*.sh

# Create output directories
RUN mkdir -p outputs/{checkpoints,logs,eval} data/{raw,processed}

# ---- Environment variables (overridable at runtime) ----------------------
ENV PYTHONPATH=/app \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/models

# ---- Default command: interactive shell (override for training/eval) ------
# Training:  docker run ... python src/training/train.py --config configs/training.yaml
# Eval:      docker run ... python src/evaluation/evaluate.py --model_path outputs/checkpoints/final
CMD ["bash"]
