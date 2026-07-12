# Build from the repo root: docker build -f backend/Dockerfile .
# (needs access to router/ which lives at the repo root, shared with the
# Track 1 submission image)
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Same as the Track 1 image: the prebuilt CPU wheel index tops out at a
# version that segfaults on Qwen2 inference (verified empirically) -
# compile a current version from source instead.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir llama-cpp-python

# GGUF weights downloaded at build time (not committed to git) and
# byte-size verified, same as the Track 1 image's Dockerfile.
RUN mkdir -p models && \
    curl -L -o models/qwen3-4b-instruct-2507-q4_k_m.gguf \
      "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" && \
    [ "$(stat -c%s models/qwen3-4b-instruct-2507-q4_k_m.gguf)" = "2497281120" ]

COPY router/ ./router/
COPY backend/main.py .

# Cloud Run injects $PORT (defaults to 8080) and expects the container to
# listen on it.
ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
