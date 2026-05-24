# Edge Audio Framework

A modular, offline-first AI framework for real-time audio analysis on headless edge devices. Runs 14 parallel audio intelligence tasks using state-of-the-art models — all locally, with zero cloud dependency.

---

## Features

| # | Task | Model | Size | Speed |
|---|------|-------|------|-------|
| 1 | **Voice Activity Detection** | Silero VAD | 5 MB | < 1ms |
| 2 | **Speech-to-Text (ASR)** | Whisper base (CTranslate2 INT8) | 74 MB | ~1s |
| 3 | **Keyword Spotting** | MIT AST Speech Commands v2 | 350 MB | ~10s |
| 4 | **Speaker Identification** | SpeechBrain ECAPA-TDNN (VoxCeleb) | 80 MB | ~1s |
| 5 | **Emotion Detection** | SUPERB HuBERT-base | 90 MB | ~2s |
| 6 | **Speech Quality (DNSMOS)** | Custom DSP + SNR estimator | 0 MB | < 1ms |
| 7 | **Language Identification** | SpeechBrain VoxLingua107 ECAPA | 80 MB | ~0.5s |
| 8 | **Environmental Sound (ESC)** | MIT AST AudioSet | 350 MB | ~2s |
| 9 | **Acoustic Event Detection** | PANNs MobileNetV2 / AST fallback | 14 MB | ~8ms |
| 10 | **Audio Quality Monitor** | Custom DSP (MOS, SNR, clipping) | 0 MB | < 1ms |
| 11 | **Music / Speech Detection** | Zero-crossing + spectral features | 0 MB | < 1ms |
| 12 | **Anomaly Detection** | Statistical outlier (autoencoder) | 0 MB | ~3s |
| 13 | **Impulse Event Detector** | Crest factor + transient analysis | 0 MB | < 1ms |
| 14 | **Music Genre Classification** | DistilHuBERT GTZAN | 90 MB | ~2s |

All models are cached inside the project's `models/` folder for full portability.

### Key Enhancements

* **Dynamic VAD Resolver**: Resolves Silero VAD local ONNX directories dynamically (handling both v4 and v5 path structures) to guarantee fully offline, zero-dependency model initialization on both Windows and Linux devices.
* **Full Multilingual ASR**: Transcribes multilingual speech completely without truncation or size restrictions, mapping ISO 639-1 language codes (e.g., `es` -> `Spanish`, `te` -> `Telugu`) to their full names.

---

## Project Structure

```
edge_audio_framework/
  core/                   # Framework engine (registry, pipeline, audio I/O)
  tasks/                  # 14 independent AI task modules
  agent/                  # Agentic decision engine
  agent_memory/           # Persistent event memory (JSONL)
  models/                 # All AI model weights (auto-cached here)
  fast_run.py             # Live microphone analysis (parallel)
  fast_run_file.py        # Audio file analysis (parallel)
  download_models.py      # One-time model downloader
  requirements.txt        # Python dependencies
```

---

## Setup & Installation

### Step 1: Clone the Repository

```bash
git clone <your-repo-url> edge_audio_framework
cd edge_audio_framework
```

---

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install System Dependencies (Linux Only)

Linux requires native audio libraries for microphone access and WAV processing:

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 libsndfile1 python3-pyaudio ffmpeg
```

> Windows users do not need this step — PyAudio ships pre-built.

---

### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5: Download AI Models

This downloads all 9 AI models (~1.2 GB total) into the local `models/` directory:

```bash
python download_models.py
```

> **Offline deployment:** If you are setting up on a machine without internet, copy the entire `models/` folder from a previously set up machine via USB or network share. The framework will detect and use the local models automatically.

---

## Usage

### Option A: Live Microphone Analysis

Records audio from your microphone and runs AI tasks in real-time.

**Run all default tasks:**
```bash
python fast_run.py --agent
```

**Run specific tasks (recommended for low-RAM devices):**

```bash
# Batch 1: Speech & Keyword Detection
python fast_run.py --agent --tasks vad,asr,keyword_spotting

# Batch 2: Identity, Emotion & Quality
python fast_run.py --agent --tasks speaker_id,emotion,speech_quality

# Batch 3: Language & Environmental Audio
python fast_run.py --agent --tasks lang_accent_id,esc,music_genre

# Batch 4: Signal Processing & Anomalies (ultra-fast, no AI models)
python fast_run.py --agent --tasks audio_quality_monitoring,anomaly_detection,impulse_event
```

**Additional options:**
```bash
python fast_run.py --duration 10            # Record 10 seconds instead of 5
python fast_run.py --device 1               # Use a specific microphone device
python fast_run.py --list-devices           # List all available microphone devices
python fast_run.py --no-prewarm             # Skip model pre-loading (faster startup)
python fast_run.py --calibrate              # Record 2s ambient baseline and calibrate thresholds
python fast_run.py --investigations         # Print a formatted report of all investigations and exit
```

---

### Option C: Simulation Demo (No Microphone Required)

Runs a complete, self-contained simulation demonstrating the entire agentic architecture: environment calibration, distress-triggered alerts, multi-turn follow-up investigations (+10s, +30s, +60s), and watchdog report generations.

```bash
python demo.py
```

---

### Option B: Audio File Analysis

Analyze a pre-recorded `.wav` file instead of using a live microphone. Ideal for headless/remote edge devices.

```bash
# Analyze a WAV file with all default tasks
python fast_run_file.py --file "path/to/audio.wav" --agent

# Analyze with specific tasks
python fast_run_file.py --file "path/to/audio.wav" --agent --tasks vad,asr,emotion

# Example (Windows)
python fast_run_file.py --file "C:\recordings\sample.wav" --agent --tasks vad,asr,lang_accent_id

# Example (Linux)
python fast_run_file.py --file "/home/user/recordings/sample.wav" --agent --tasks vad,asr,lang_accent_id
```

---

## Available Task Names

Use these exact names with the `--tasks` flag (comma-separated, no spaces):

| Task Name | Description |
|-----------|-------------|
| `vad` | Voice Activity Detection |
| `asr` | Automatic Speech Recognition (Whisper) |
| `keyword_spotting` | Keyword / Command Detection |
| `speaker_id` | Speaker Identification & Counting |
| `emotion` | Emotion Recognition |
| `speech_quality` | Speech Quality (DNSMOS + SNR) |
| `lang_accent_id` | Language & Accent Identification |
| `esc` | Environmental Sound Classification |
| `acoustic_event_detection` | Acoustic Event Detection |
| `audio_quality_monitoring` | Audio Quality Monitor (MOS, SNR, Clipping) |
| `music_speech_detection` | Music vs. Speech Classifier |
| `anomaly_detection` | Audio Anomaly Detector |
| `impulse_event` | Impulse / Gunshot Event Detector |
| `music_genre` | Music Genre Classification |

---

## Offline / Air-Gapped Deployment

The framework is designed for fully offline operation after initial setup:

1. **Set up on an internet-connected machine** following the steps above.
2. **Copy the entire `edge_audio_framework/` folder** (including `models/`) to the target device via USB.
3. **On the target device**, create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Run strictly offline** by setting the environment variable:
   ```bash
   # Linux
   export TRANSFORMERS_OFFLINE=1
   python fast_run.py --agent --tasks vad,asr,emotion

   # Windows (PowerShell)
   $env:TRANSFORMERS_OFFLINE="1"
   python fast_run.py --agent --tasks vad,asr,emotion
   ```

> All model weights are stored in the local `models/` directory. No internet is required at runtime.

---

## Agent Decision Engine

When you run with `--agent`, the framework deploys an advanced **3-Agent Orchestration Architecture** to reason about parallel task outputs:

1. **Triage Agent** (`agent/triage_agent.py`): Dynamically schedules and filters which tasks to run based on active policy profiles (Balanced, Industrial Safety, Privacy First, Low Power) and recent historical event trends (e.g., repeated noise anomalies, hardware quality degradation).
2. **Synthesis Agent** (`agent/synthesis_agent.py`): Consumes all raw parallel task results, applies structured Chain-of-Thought (CoT) reasoning, and produces a single unified decision payload (with event classification, priority levels, and detailed operator-facing incident reports).
3. **Watchdog Agent** (`agent/watchdog_agent.py`): Records decisions in persistent cross-session memory (`agent_memory/events.jsonl`), detects pattern/risk escalations over lookback windows, monitors microphone clipping/SNR, and generates diagnostic watchdog reports.

### Advanced Agentic Capabilities

* **Self-Calibrating Agent** (`agent/calibration_agent.py`): Automatically calibrates the system on startup by recording a 2-second baseline window of ambient environment noise. It classifies the location (e.g. `silent_room`, `office`, `outdoor`, `industrial`, or `machinery_heavy`), configures tailored initial detection thresholds (VAD speech ratios, SNR alerts, anomaly scores, energy spike limits), and dynamically adapts these parameters over time using a false-positive tracking feedback loop.
* **Multi-Turn Incident Investigation** (`agent/investigation_agent.py`): For any critical HIGH-priority alert (e.g. distress trigger or impulse threat), the orchestrator automatically launches a background multi-turn investigation loop. It schedules follow-up ambient recordings at +10s, +30s, and +60s offsets, builds a comparative risk timeline, and reaches a final threat verdict (`threat_confirmed`, `threat_resolved`, `false_alarm`, or `inconclusive`).

### Resilience & Fallbacks
If any agent in the 3-agent orchestration pipeline raises an exception or fails, the orchestrator automatically degrades gracefully to the original deterministic, rule-based decision engine (`agent/audio_agent_rules.py`). This prevents crashes on critical edge deployments.

All processing occurs completely locally on the edge device. No audio or transcript data is transmitted to the cloud.

---

## Troubleshooting

### Microphone not detected
```bash
python fast_run.py --list-devices
# Then specify the correct device ID:
python fast_run.py --device 2 --agent --tasks vad,asr
```

### Models downloading every time (Windows)
This happens if Windows blocks symbolic links. The framework includes automatic symlink-to-copy patches, but if issues persist:
```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
python fast_run.py --agent
```

### Out of Memory on low-RAM devices
Run tasks in smaller batches (2-3 tasks at a time) instead of all at once:
```bash
python fast_run.py --agent --tasks vad,asr
```

### Linux: PyAudio installation fails
```bash
sudo apt-get install -y portaudio19-dev python3-dev
pip install pyaudio
```

---

## Platform Compatibility

| Feature | Windows 10/11 | Ubuntu 20.04+ | Raspberry Pi OS |
|---------|:------------:|:------------:|:--------------:|
| Live Microphone | Yes | Yes | Yes |
| File Analysis | Yes | Yes | Yes |
| GPU Acceleration | CUDA (if available) | CUDA (if available) | CPU only |
| Offline Mode | Yes | Yes | Yes |

---

## License

Internal use only. All AI models are subject to their respective open-source licenses.
