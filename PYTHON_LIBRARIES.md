# Python Libraries Reference — Edge Audio Framework

This document lists all Python standard and external libraries utilized in the Edge Audio Framework, explaining their specific purpose and active role in the codebase.

---

## 1. Deep Learning & Core AI Frameworks

| Library Name | Version | Core Purpose | Live Role in Framework |
| :--- | :--- | :--- | :--- |
| **`torch`** (PyTorch) | `>=2.0.0` | Tensor computation and neural networks | Provides the deep learning backend for running inference on speaker recognition, language identification, and emotion classifiers. |
| **`torchaudio`** | `>=2.0.0` | Audio signal processing extensions | Reads, writes, resamples, and transforms raw audio waveforms into PyTorch tensors. |
| **`transformers`** | `>=4.30.0` | Hugging Face transformer models | Loads neural network configurations, feature extractors, and processor tokenizers for the acoustic feature models. |
| **`accelerate`** | `>=0.20.0` | Hardware memory orchestration | Speeds up weights loading and optimizes tensor allocations on CPU and GPU devices. |
| **`safetensors`** | `>=0.3.0` | Secure weight serialization format | Safely loads pre-trained model weights locally (preventing code execution exploits associated with standard pickle formats). |

---

## 2. Speech, ASR & Audio Processing

| Library Name | Version | Core Purpose | Live Role in Framework |
| :--- | :--- | :--- | :--- |
| **`speechbrain`** | `>=0.5.14` | Speech classification recipes | Orchestrates the Speaker Recognition (`spkrec-ecapa-voxceleb`) and Language Identification (`lang-id-voxlingua107-ecapa`) modules. |
| **`faster-whisper`** | `>=1.0.0` | Optimized speech-to-text | Performs high-speed multilingual Automatic Speech Recognition (ASR) using CTranslate2. |
| **`ctranslate2`** | `>=4.0.0` | High-performance inference engine | Backend execution accelerator that compiles and runs Whisper weights efficiently in C++. |
| **`onnxruntime`** | `>=1.15.0` | ONNX execution engine | Loads and runs the Silero Voice Activity Detector (VAD) model (`silero_vad.onnx`) locally and offline. |
| **`librosa`** | `>=0.10.0` | Audio and music analysis toolkit | Computes acoustic features (such as MFCCs and spectrograms) and measures Signal-to-Noise Ratio (SNR) for quality checks. |
| **`soundfile`** | `>=0.12.0` | Audio file IO | Reads and writes standard audio files (WAV, FLAC, OGG) to and from NumPy arrays. |
| **sounddevice** | `>=0.4.0` | PortAudio bindings | Connects to hardware audio drivers to play and capture audio signals. |
| **PyAudio** | `>=0.2.14` | Direct audio stream interface | Feeds real-time audio samples from system microphones into the framework during live recording runs. |

---

## 3. Data Science & Document Utilities

| Library Name | Version | Core Purpose | Live Role in Framework |
| :--- | :--- | :--- | :--- |
| **`numpy`** | `>=1.24.0` | Multi-dimensional array operations | Processes audio waveform buffers, root-mean-square amplitude, and baseline feature arrays. |
| **`scipy`** | `>=1.10.0` | Scientific signal filtering | Applies signal processing filters (low-pass, high-pass) and computes signal statistics. |
| **`pandas`** | `>=2.0.0` | Tabular data manipulation | Used to manage datasets, log structured statistics, and organize test results. |
| **`scikit-learn`** | `>=1.2.0` | Machine learning toolkit | Used by the `CalibrationAgent` to scale features, normalize signals, and classify baseline ambient environments. |
| **`pyyaml`** | `>=6.0` | YAML configuration parser | Parses core application configuration settings and parameter schemas. |
| **`HyperPyYAML`** | `>=1.2.0` | Extended YAML parser | Parses extended YAML recipes with object instantiation instructions (specifically required by SpeechBrain). |
| **`huggingface_hub`**| `>=0.20.0`| Offline repository manager | Manages weight caching directories and ensures local-only folder resolving for offline models. |
| **`tokenizers`** | `>=0.13.0`| Sub-word text tokenization | Decodes speech-to-text token probability distributions back into plain characters. |
| **`sentencepiece`** | `>=0.1.99`| Unsupervised tokenization | Processes sub-word translations and token mappings for SpeechBrain architectures. |
| **`python-docx`** | `>=1.0.0` | Word document generation | Compiles formatted Markdown manuals and setup files into Microsoft Word (`.docx`) format. |

---

## 4. Standard Library Modules (Core Built-ins)

| Module Name | Type | Core Purpose | Live Role in Framework |
| :--- | :--- | :--- | :--- |
| **`pathlib`** | Built-in | Path object navigation | Resolves path references relative to the active file (`Path(__file__).parent`) across Windows and Linux. |
| **`shutil`** | Built-in | System folder operations | Stages extracted files, handles renames, and removes temporary models folders securely. |
| **`zipfile`** | Built-in | Archive management | Inspects zip file headers (`is_zipfile`) and extracts packed weight files. |
| **`threading`** | Built-in | Asynchronous thread runtime | Runs background daemon threads to prewarm heavy models in parallel with hardware initialization. |
| **`datetime`** / **`time`**| Built-in | Date and epoch time clocks | Tracks duration thresholds, calibration intervals, and logs timing metadata. |
| **`re`** | Built-in | Regular expressions | Matches filename strings (e.g. part number extraction) and processes inline formatting for docs conversion. |
| **`json`** | Built-in | JSON serialization | Reads and writes environmental configuration keys and logs multi-turn incident investigations. |
| **`unittest`** | Built-in | Test runner framework | Performs automated regression tests for Triaging, Synthesis, and Watchdog agents. |
