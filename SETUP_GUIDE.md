# Edge Audio Framework — Setup and Installation Guide

This guide provides step-by-step instructions for installing, configuring, calibrating, and running the Edge Audio Framework on Windows and Linux systems.

---

## 1. Prerequisites

Before installing the Python dependencies, ensure your system has the required compiler and audio libraries installed.

### Windows Requirements
- **Python 3.9, 3.10, or 3.11** (recommended; ensure it is added to your system PATH).
- **Microsoft C++ Build Tools** (required to compile PyAudio and other C++ dependencies if pre-compiled wheels are not found).

### Linux Requirements (Debian/Ubuntu)
Install Python development headers, portaudio library, and compiler tools:
```bash
sudo apt-get update
sudo apt-get install -y python3-dev portaudio19-dev gcc g++ make
```

---

## 2. Step-by-Step Installation

Follow these steps to set up the virtual environment and install the required modules.

### Step A: Clone the Repository & Navigate to Folder
Open your terminal or PowerShell and change directories to the project root:
```bash
cd f:/edge_audio_framework
```

### Step B: Create a Virtual Environment
Create a clean, isolated virtual environment to prevent package version conflicts:
```bash
python -m venv venv
```

### Step C: Activate the Virtual Environment
Activate the environment to ensure Python uses the local virtual environment packages:
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\activate
  ```
- **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
- **Linux / macOS**:
  ```bash
  source venv/bin/activate
  ```

### Step D: Install Core Dependencies
Install the required packages using the `requirements.txt` file:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> If you encounter compilation errors for `PyAudio` on Windows, download the pre-compiled `.whl` file matching your Python version from the Christoph Gohlke archives or use `pip install pipwin` followed by `pipwin install pyaudio`.

---

## 3. Reassemble Split AI Model Weights

Because HuggingFace weights exceed GitHub file size limits, model assets are distributed as split binary parts (`models.zip.part1`, `models.zip.part2`). You must run the reassembly tool before running the application:

```bash
python reassemble_models.py
```

### What the tool does:
1. **Scans and Sorts**: Automatically detects all split parts in the project root and sorts them.
2. **Validates Integrity**: Checks sizes and runs a zip validation check.
3. **Overwrites Safely**: Warns you if a `models/` directory already exists and asks to confirm.
4. **Stages and Swaps**: Extracts files to `models_temp/` and stages them in `models_new/` before swapping atomically into `models/` (preventing half-extracted corrupted states).
5. **Permissions Fix**: Handles Windows read-only flags inside local `.git` directories automatically.

---

## 4. Run System Self-Calibration

The agent uses ambient sound levels to adjust noise gates and thresholds (e.g. for VAD, SNR, and anomaly detection). Run calibration before your first run:

```bash
python fast_run.py --calibrate --duration 10
```

- This records ambient noise for 10 seconds.
- Analyzes and classifies your room environment (e.g. `silent_room`, `office`, `outdoor`, `industrial`).
- Saves baseline thresholds to `agent_memory/calibration.json`.

---

## 5. Verify the Installation (Tests & Simulation)

Ensure the codebase works correctly using the test suite and demo scenarios:

### Run Unit Tests
Verify all 3-Agent pipelines, ASR translation, and routing rules:
```bash
python -m unittest discover -s tests
```

### Run Simulation Demo
Verify the multi-turn agent investigations, VAD speech gates, and Watchdog trend reports using mocked scenarios:
```bash
python demo.py
```

---

## 6. Run the Application

The framework supports both live microphone recording and static file analysis:

### Live Microphone Analysis
Analyze live audio using the orchestrator with asynchronous background model loading:
```bash
python fast_run.py --agent --tasks vad,asr,emotion
```

### Audio File Analysis
Perform agentic analysis on an existing audio file:
```bash
python fast_run_file.py -f sample_data/test-1.wav --agent --tasks vad,asr,emotion
```

---

## 7. Troubleshooting

### Startup Performance (Latency)
- **First Run / Cold Load**: If `agent_memory/calibration.json` is missing, the application assumes a cold load, using a timeout of 450 seconds.
- **Warm Runs**: If calibration exists, subsequent runs will load models in background threads under a 90-second timeout, leveraging OS caches.

### Offline Mode Support
The framework runs fully offline. The global environment variables are registered in the scripts to enforce offline HuggingFace imports:
```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
```
Ensure that model weights are fully unpacked in the `models/` directory using the `reassemble_models.py` utility.
