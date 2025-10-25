FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---- System deps ----
RUN apt-get update && apt-get install -y git python3 python3-pip && \
    pip install --upgrade pip

# ---- Python deps ----
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- Workspace ----
WORKDIR /workspace
COPY train_leaf.py .
COPY hf_token.txt .     # optional fallback if env var not set

# ---- Hugging Face / Git credentials (optional) ----
# Pass tokens during docker run via env vars for safety
#   -e HF_TOKEN=hf_xxxxx
#   -e GIT_USER=you -e GIT_EMAIL=you@example.com -e GIT_PAT=ghp_xxxxx

# ---- Entry ----
CMD ["python3", "train_leaf.py"]
