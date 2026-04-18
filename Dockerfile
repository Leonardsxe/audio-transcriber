# =============================================================================
#  audio-transcriber — Multi-stage Docker build
#
#  Stage "base"  → transcription only  (faster-whisper + ffmpeg)    ~1.5 GB
#  Stage "full"  → + diarization       (pyannote.audio + torch CPU) ~4   GB
#
#  Usage:
#    docker compose build transcribe   # builds up to "base"
#    docker compose build diarize      # builds up to "full"
# =============================================================================

# ─── Stage 1: base (transcription only) ──────────────────────────────────────
FROM python:3.11-slim-bookworm AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
#   - ffmpeg: required by pydub for audio conversion
#   - libsndfile1: required by some audio libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency specification first (layer caching)
COPY pyproject.toml ./

# Install core dependencies
RUN pip install --no-cache-dir .

# Copy application source code
COPY transcriber/ ./transcriber/

# Re-install in editable-like mode so the entrypoint resolves correctly
RUN pip install --no-cache-dir -e .

# Create directories for volume mounts
RUN mkdir -p /app/audio /app/output

# Default entrypoint — the CLI
ENTRYPOINT ["python", "-m", "transcriber.main"]
CMD ["--help"]

# ─── Stage 2: full (base + diarization) ─────────────────────────────────────
FROM base AS full

# Install CPU-only torch first (avoids the 2 GB CUDA wheel)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install diarization extras
RUN pip install --no-cache-dir ".[diarization]"
