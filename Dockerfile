FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Hugging Face Spaces often mount persistent storage at /app, which hides any
# application files copied into /app in the image. Keep code in /workspace and
# point data root at /app so jobs survive restarts.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DFL_DATA_ROOT=/app \
    NN_KEEP_CUDA_VISIBLE_DEVICES=1 \
    HF_EXIT_WHEN_DONE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-hf-space.txt /workspace/requirements-hf-space.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r /workspace/requirements-hf-space.txt

COPY . /workspace

EXPOSE 7860

CMD ["sh", "-c", "python3 -m uvicorn runonhugging:app --host 0.0.0.0 --port ${PORT:-7860}"]
