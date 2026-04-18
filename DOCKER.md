# Docker Setup Guide

Run the audio-transcriber on **any machine** (Windows or Linux) without
installing Python, ffmpeg, or managing virtual environments.

---

## Prerequisites

### Linux (Ubuntu / Debian / Pop!_OS)

```bash
# 1. Install Docker Engine
sudo apt update
sudo apt install -y docker.io docker-compose-plugin

# 2. Add your user to the docker group (avoids needing sudo for every command)
sudo usermod -aG docker $USER

# 3. Log out and back in, then verify
docker --version
docker compose version
```

### Windows

1. Download and install **[Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)**
2. During installation, enable **WSL 2 backend** (recommended)
3. Restart your computer
4. Open a terminal (PowerShell or CMD) and verify:
   ```powershell
   docker --version
   docker compose version
   ```

> **Note:** On Windows, all commands below work the same in PowerShell,
> CMD, or the WSL 2 terminal.

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/audio-transcriber.git
cd audio-transcriber
```

---

## Step 2 — Configure environment

Copy the template and edit it:

```bash
# Linux / macOS / WSL
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

Open `.env` in any text editor and adjust as needed:

```env
# Default — CPU (works everywhere)
TRANSCRIBER_DEVICE=cpu
TRANSCRIBER_COMPUTE_TYPE=int8

# If the machine has an NVIDIA GPU + nvidia-container-toolkit,
# uncomment these and comment the two lines above:
# TRANSCRIBER_DEVICE=cuda
# TRANSCRIBER_COMPUTE_TYPE=float16
```

### For diarization (speaker identification)

If you need speaker identification, you also need a HuggingFace token:

1. Create a free account at https://hf.co
2. Accept model terms at:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
3. Create a read token at https://hf.co/settings/tokens
4. Add it to `.env`:
   ```env
   HF_TOKEN=hf_your_token_here
   ```

---

## Step 3 — Build the Docker image

Choose which image to build based on your needs:

```bash
# Transcription only (smaller image, ~1.5 GB + model download)
docker compose build transcribe

# Full pipeline with speaker diarization (~4 GB + model download)
docker compose build diarize
```

> **First build** takes several minutes to download and install dependencies.
> Subsequent builds are fast thanks to Docker layer caching.

---

## Step 4 — Place your audio files

Put your audio files (`.mp3`, `.m4a`, `.wav`, `.ogg`, `.flac`, etc.) into
the `audio/` directory:

```bash
mkdir -p audio
# Copy or move your files into audio/
cp /path/to/interview.m4a audio/
```

On Windows:
```powershell
mkdir audio -Force
Copy-Item C:\path\to\interview.m4a audio\
```

---

## Step 5 — Run transcription

### Transcribe a single file

```bash
docker compose run --rm transcribe audio/interview.m4a --chunk
```

### Transcribe with SRT subtitle export

```bash
docker compose run --rm transcribe audio/interview.m4a --chunk --format srt
```

### Transcribe with speaker diarization

```bash
docker compose run --rm diarize audio/interview.m4a \
    --chunk --chunk-minutes 10 \
    --diarize --diarize-format transcript.json
```

### Human-readable diarized output (for NVivo / ATLAS.ti)

```bash
docker compose run --rm diarize audio/interview.m4a \
    --chunk --chunk-minutes 10 \
    --diarize --diarize-format transcript.txt
```

### Batch transcribe all files in a directory

```bash
docker compose run --rm transcribe audio/ --batch --chunk --format json
```

> **Results** are saved to the `output/` directory on your host machine.

---

## Step 6 — Check your results

```bash
ls output/
# interview.txt  interview.json  interview.srt  etc.
```

---

## CLI Reference (quick)

All CLI options from the regular project work inside Docker:

```
docker compose run --rm transcribe [AUDIO_FILE] [OPTIONS]

Options:
  --format {txt,json,srt}           Raw transcript export format
  --output-dir DIR                  Output directory (default: ./output)
  --chunk                           Split long audio at silence boundaries
  --chunk-minutes N                 Target chunk length in minutes (default: 10)
  --diarize                         Run speaker diarization (needs HF_TOKEN)
  --diarize-format {transcript.json,transcript.txt}
  --flip-roles                      Swap INTERVIEWER / INTERVIEWEE
  --batch                           Transcribe all files in a directory
  --log-level {DEBUG,INFO,WARNING,ERROR}
```

---

## Model Caching

The Whisper model (`large-v3` ≈ 3 GB) is downloaded on first use and stored
in a Docker volume called `audio-transcriber-whisper-cache`. This means:

- **First run** takes longer (model download)
- **Subsequent runs** start immediately (model is cached)
- The cache persists even if you remove and rebuild the container

To clear the cache and force a re-download:
```bash
docker volume rm audio-transcriber-whisper-cache
```

---

## GPU Support (Optional)

If the host machine has an **NVIDIA GPU** and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
is installed:

1. Update `.env`:
   ```env
   TRANSCRIBER_DEVICE=cuda
   TRANSCRIBER_COMPUTE_TYPE=float16
   ```

2. Add GPU access to `docker-compose.yml` under the service you want to use:
   ```yaml
   diarize:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

3. Rebuild and run:
   ```bash
   docker compose build diarize
   docker compose run --rm diarize audio/interview.m4a --chunk --diarize
   ```

---

## Troubleshooting

### `docker: command not found`

Docker is not installed. Follow the Prerequisites section above.

### `permission denied while trying to connect to the Docker daemon`

On Linux, your user needs to be in the `docker` group:
```bash
sudo usermod -aG docker $USER
# Then log out and back in
```

### Build fails with `pip install` errors

Make sure you have a stable internet connection. Docker needs to download
Python packages during the build. If behind a proxy, configure Docker's
proxy settings.

### `No audio files found`

Make sure your audio files are in the `audio/` directory relative to the
project root, not in a subdirectory.

### Model download is slow

The `large-v3` model is ~3 GB. On slow connections, consider using a
smaller model by setting in `.env`:
```env
TRANSCRIBER_MODEL_SIZE=medium
```

### Container runs out of memory

The `large-v3` model needs ~10 GB RAM. If Docker is constrained:
- **Windows (Docker Desktop)**: Settings → Resources → increase memory to 12 GB
- **Linux**: Usually not an issue (Docker uses host memory directly)
- Or use a smaller model: `TRANSCRIBER_MODEL_SIZE=medium` (~5 GB RAM)
