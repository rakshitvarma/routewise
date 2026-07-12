FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# The prebuilt CPU wheel index tops out at 0.2.62, which predates full
# Qwen2 architecture support in llama.cpp and segfaults on inference
# (verified empirically - loads fine, crashes on first generate call).
# Compile a current version from source instead.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir llama-cpp-python

# GGUF weights are downloaded at build time rather than committed to git
# (over GitHub's practical limits, and the repo needs to stay
# clonable/buildable from source alone per the submission guide's
# "runnable using the provided instructions" requirement). Byte-size
# verified against Hugging Face's reported Content-Length - a previous
# manual download silently truncated (~174MB short) and loaded fine but
# segfaulted on first inference, which is why this checks explicitly
# rather than trusting curl's exit code alone.
# Placed before the code COPY so a code change doesn't re-download it.
#
# Single consolidated model (Qwen3-4B-Instruct-2507) replacing the earlier
# two separate 1.5B general/code models: a newer architecture generation,
# handles code well enough on its own to drop the dedicated coder model,
# and at ~2.5GB uses less combined memory than running two 1.5B models
# would if both were upgraded to 3B instead.
RUN mkdir -p models && \
    curl -L -o models/qwen3-4b-instruct-2507-q4_k_m.gguf \
      "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" && \
    [ "$(stat -c%s models/qwen3-4b-instruct-2507-q4_k_m.gguf)" = "2497281120" ]

# The harness mounts /input and /output; ensure they exist so a missing
# mount doesn't crash us before we even read the tasks file.
RUN mkdir -p /input /output

COPY router/ ./router/
COPY main.py .

ENTRYPOINT ["python", "main.py"]
