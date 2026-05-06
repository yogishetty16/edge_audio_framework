"""
run_live.py
===========
USE THIS INSTEAD OF live_test.py ON YOUR COMPANY LAPTOP.
It applies the SSL corporate proxy fix automatically before running.

Usage:
    python run_live.py                         -- 5 sec, default tasks
    python run_live.py --duration 10           -- 10 seconds
    python run_live.py --tasks all             -- all 12 tasks
    python run_live.py --tasks vad,asr,emotion -- specific tasks
    python run_live.py --device 1              -- specific mic device
"""

# ── STEP 1: Apply SSL fix BEFORE any other import ──────────────────────────
import ssl
import os
import sys
import warnings
import logging
import shutil

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Monkeypatch os.symlink to copy files instead. This completely bypasses the Windows 
# "Privilege Not Held" (WinError 1314) issue without triggering huggingface_hub bugs.
_orig_symlink = getattr(os, "symlink", None)
def _mock_symlink(src, dst, target_is_directory=False, dir_fd=None):
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
os.symlink = _mock_symlink

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"]           = "0"
os.environ["CURL_CA_BUNDLE"]              = ""
os.environ["REQUESTS_CA_BUNDLE"]          = ""
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # Fix Protobuf TypeError
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ["HF_HUB_VERBOSITY"]           = "error"
os.environ["TRANSFORMERS_VERBOSITY"]      = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]       = "0"

# ── Robust SSL Bypass for Corporate Proxy ───────────────────────────────────
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

try:
    import requests
    _orig_req = requests.Session.request
    def _patched_req(self, *args, **kwargs):
        kwargs["verify"] = False
        return _orig_req(self, *args, **kwargs)
    requests.Session.request = _patched_req
except Exception:
    pass

try:
    import httpx
    _orig_httpx = httpx.Client.__init__
    def _patched_httpx(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_httpx(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_httpx
except Exception:
    pass



# Set model paths
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"]    = os.path.join(_ROOT, "models", "hf_cache")
os.environ["TORCH_HOME"] = os.path.join(_ROOT, "models", "torch_hub")
sys.path.insert(0, _ROOT)

# ── STEP 2: Now safe to import framework ────────────────────────────────────
import time
import argparse
import numpy as np

def record_audio(duration_sec=5, device_index=None, sample_rate=16000):
    import pyaudio
    pa = pyaudio.PyAudio()
    chunk = 1024
    stream = pa.open(rate=sample_rate, channels=1, format=pyaudio.paFloat32,
                     input=True, frames_per_buffer=chunk,
                     input_device_index=device_index)
    print(f"\nRecording {duration_sec}s ... speak now!")
    for i in range(3, 0, -1):
        print(f"  {i}..."); time.sleep(1)
    print("  RECORDING NOW")
    frames = []
    for _ in range(int(sample_rate / chunk * duration_sec)):
        frames.append(stream.read(chunk, exception_on_overflow=False))
    stream.stop_stream(); stream.close(); pa.terminate()
    data = np.frombuffer(b"".join(frames), dtype=np.float32)
    print(f"  Recorded {len(data)/sample_rate:.1f}s\n")
    return data, sample_rate


def run_analysis(duration, tasks, device):
    from core.audio_io import AudioData
    from core.pipeline import AudioPipeline

    audio_np, sr = record_audio(duration_sec=duration, device_index=device)
    audio = AudioData(waveform=audio_np, sample_rate=sr,
                      duration_sec=len(audio_np)/sr, n_channels=1)

    print("Running analysis...")
    pipeline = AudioPipeline(tasks=tasks)
    t0 = time.perf_counter()
    results = pipeline.run(audio)
    elapsed = (time.perf_counter() - t0) * 1000

    LABELS = {
        "vad": "Voice Activity Detection",
        "asr": "Automatic Speech Recognition",
        "keyword_spotting": "Keyword Spotting",
        "speaker_id": "Speaker ID",
        "emotion": "Emotion Detection",
        "speech_quality": "Speech Quality",
        "lang_accent_id": "Language / Accent ID",
        "esc": "Environmental Sound Class.",
        "acoustic_event_detection": "Acoustic Event Detection",
        "audio_quality_monitoring": "Audio Quality Monitor",
        "music_speech_detection": "Music/Speech Detection",
        "anomaly_detection": "Anomaly Detection",
    }

    print(f"\n{'='*55}\n  RESULTS  (pipeline: {elapsed:.0f}ms)\n{'='*55}")

    for task_name, r in results.items():
        if task_name.startswith("_") or r is None:
            continue
        label = LABELS.get(task_name, task_name)
        if not r.get("success", True):
            print(f"\n  XX  {label}: {r.get('error','?')}")
            continue
        ms = f"[{r['latency_ms']:.0f}ms]" if r.get("latency_ms") else ""
        print(f"\n  OK  {label}  {ms}")

        if task_name == "vad":
            print(f"      Speech ratio: {r.get('speech_ratio',0)*100:.1f}%  "
                  f"| Total speech: {r.get('total_speech_sec',0):.2f}s")
        elif task_name == "asr":
            print(f"      Transcript: \"{r.get('text','')[:120]}\"")
            print(f"      Language:   {r.get('language','N/A')}")
        elif task_name == "keyword_spotting":
            print(f"      Top keyword: {r.get('top_label','')} "
                  f"({r.get('top_score',0)*100:.1f}%)")
        elif task_name == "speaker_id":
            print(f"      Speakers: {r.get('num_speakers',0)}")
        elif task_name == "emotion":
            print(f"      Emotion: {r.get('top_emotion','')} "
                  f"({r.get('top_score',0)*100:.1f}%)")
            if r.get("sentiment"):
                print(f"      Sentiment: {r['sentiment']}")
        elif task_name == "speech_quality":
            print(f"      DNSMOS: {r.get('dnsmos_overall',0):.2f}/5  "
                  f"| SNR: {r.get('snr_db',0):.1f}dB")
        elif task_name == "lang_accent_id":
            print(f"      Language: {r.get('top_language','')} "
                  f"({r.get('top_language_score',0)*100:.1f}%)")
            if r.get("top_accent"):
                print(f"      Accent: {r['top_accent']}")
        elif task_name == "esc":
            print(f"      Sound: {r.get('top_class','')} "
                  f"({r.get('top_score',0)*100:.1f}%)")
        elif task_name == "acoustic_event_detection":
            for ev in (r.get("events") or [])[:3]:
                print(f"        - {ev.get('label','')} "
                      f"({ev.get('score',0)*100:.1f}%)")
        elif task_name == "audio_quality_monitoring":
            print(f"      Quality: {r.get('quality_label','').upper()}  "
                  f"| MOS: {r.get('mos',0):.2f}  "
                  f"| SNR: {r.get('snr_db',0):.1f}dB  "
                  f"| Clipping: {r.get('clipping_detected',False)}")
        elif task_name == "music_speech_detection":
            print(f"      Speech: {r.get('speech_fraction',0)*100:.1f}%  "
                  f"| Music: {r.get('music_fraction',0)*100:.1f}%  "
                  f"| Noise: {r.get('noise_fraction',0)*100:.1f}%")
        elif task_name == "anomaly_detection":
            flag = "ANOMALY!" if r.get("is_anomaly") else "Normal"
            print(f"      Status: {flag}  "
                  f"| Score: {r.get('anomaly_score',0):.4f}")

    print(f"\n{'='*55}\nDone!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Live Audio Test (with SSL corporate proxy fix)")
    parser.add_argument("--duration", "-d", type=int, default=5,
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument("--device", type=int, default=None,
                        help="Mic device index. Run with --list-devices to see options")
    parser.add_argument("--tasks", "-t",
                        default="vad,asr,emotion,keyword_spotting,speech_quality,audio_quality_monitoring",
                        help="Comma-separated tasks or 'all'")
    parser.add_argument("--list-devices", action="store_true",
                        help="List all available microphone devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        import pyaudio
        pa = pyaudio.PyAudio()
        print("\nAvailable microphone devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']}")
        pa.terminate()
        return

    ALL = ["vad","asr","keyword_spotting","speaker_id","emotion","speech_quality",
           "lang_accent_id","esc","acoustic_event_detection",
           "audio_quality_monitoring","music_speech_detection","anomaly_detection"]
    task_list = ALL if args.tasks.lower() == "all" else \
                [t.strip() for t in args.tasks.split(",")]

    print("=" * 55)
    print("  Edge Audio Framework - Live Test")
    print("  [SSL corporate proxy fix is active]")
    print("=" * 55)
    print(f"  Duration : {args.duration}s")
    print(f"  Tasks    : {', '.join(task_list)}")

    run_analysis(args.duration, task_list, args.device)


if __name__ == "__main__":
    main()
