"""
wake_word_listener.py
=====================
Continuously listens to the microphone.
When it hears "LTTS" (e.g. "Hi LTTS", "Hey LTTS", "LTTS analysis"),
it activates and runs the full audio analysis pipeline.

Usage:
    python wake_word_listener.py
    python wake_word_listener.py --wake-word LTTS
    python wake_word_listener.py --tasks vad,asr,emotion,speech_quality
    python wake_word_listener.py --duration 8 --device 1
"""

# ── SSL fix FIRST (for corporate proxy) ─────────────────────────────────────
import ssl, os, sys
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"]               = "0"
os.environ["CURL_CA_BUNDLE"]                  = ""
os.environ["REQUESTS_CA_BUNDLE"]              = ""
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # Fix Protobuf TypeError
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]          = "false"
os.environ["HF_HUB_VERBOSITY"]               = "error"

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"]    = os.path.join(_ROOT, "models", "hf_cache")
os.environ["TORCH_HOME"] = os.path.join(_ROOT, "models", "torch_hub")
sys.path.insert(0, _ROOT)

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



# ── Imports ──────────────────────────────────────────────────────────────────
import time
import argparse
import threading
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
WAKE_CHUNK_SEC   = 2      # seconds to listen for wake word at a time
ANALYSIS_SEC     = 8      # seconds to record for full analysis after wake
WAKE_OVERLAP_SEC = 0.5    # overlap between wake-word windows

ALL_TASKS = [
    "vad", "asr", "keyword_spotting", "speaker_id", "emotion",
    "speech_quality", "lang_accent_id", "esc", "acoustic_event_detection",
    "audio_quality_monitoring", "music_speech_detection", "anomaly_detection",
]

DEFAULT_TASKS = [
    "vad", "asr", "emotion", "keyword_spotting",
    "speech_quality", "audio_quality_monitoring",
]


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _open_stream(device_index=None):
    import pyaudio
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=SAMPLE_RATE, channels=1,
        format=__import__("pyaudio").paFloat32,
        input=True, frames_per_buffer=1024,
        input_device_index=device_index,
    )
    return pa, stream


def _record_seconds(stream, duration_sec):
    """Record exactly duration_sec seconds from an open stream."""
    chunk = 1024
    frames = []
    n = int(SAMPLE_RATE / chunk * duration_sec)
    for _ in range(n):
        frames.append(stream.read(chunk, exception_on_overflow=False))
    return np.frombuffer(b"".join(frames), dtype=np.float32)


# ── Wake word detection ───────────────────────────────────────────────────────

def contains_wake_word(audio_np: np.ndarray, wake_word: str) -> tuple[bool, str]:
    """
    Run lightweight ASR on a short audio clip.
    Returns (triggered, transcript).
    """
    try:
        from faster_whisper import WhisperModel
        # Load tiny model — cached after first call
        if not hasattr(contains_wake_word, "_model"):
            contains_wake_word._model = WhisperModel(
                "tiny.en", device="cpu", compute_type="int8",
                download_root=os.path.join(_ROOT, "models"),
            )
        model = contains_wake_word._model
        # Write to temp buffer
        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio_np, SAMPLE_RATE)
        segs, _ = model.transcribe(tmp_path, language="en", beam_size=1,
                                   vad_filter=True)
        transcript = " ".join(s.text for s in segs).strip()
        os.unlink(tmp_path)
        ww = wake_word.upper()
        triggered = ww in transcript.upper()
        return triggered, transcript
    except Exception as e:
        return False, ""


# ── Full analysis ─────────────────────────────────────────────────────────────

def run_full_analysis(audio_np: np.ndarray, tasks: list):
    """Run the selected pipeline tasks on audio and print results."""
    from core.audio_io import AudioData
    from core.pipeline import AudioPipeline

    audio = AudioData(
        waveform=audio_np,
        sample_rate=SAMPLE_RATE,
        duration_sec=len(audio_np) / SAMPLE_RATE,
        n_channels=1,
    )

    print("\n" + "="*55)
    print("  LTTS ACTIVATED — Running Analysis...")
    print("="*55)

    pipeline = AudioPipeline(tasks=tasks)
    t0 = time.perf_counter()
    results = pipeline.run(audio)
    elapsed = (time.perf_counter() - t0) * 1000

    LABELS = {
        "vad":                    "Voice Activity Detection",
        "asr":                    "Speech Recognition",
        "keyword_spotting":       "Keyword Spotting",
        "speaker_id":             "Speaker ID",
        "emotion":                "Emotion Detection",
        "speech_quality":         "Speech Quality",
        "lang_accent_id":         "Language / Accent ID",
        "esc":                    "Environmental Sound",
        "acoustic_event_detection": "Acoustic Events",
        "audio_quality_monitoring": "Audio Quality",
        "music_speech_detection": "Music / Speech",
        "anomaly_detection":      "Anomaly Detection",
    }

    print(f"\n  Pipeline time: {elapsed:.0f}ms\n")

    for task_name, r in results.items():
        if task_name.startswith("_") or r is None:
            continue
        label = LABELS.get(task_name, task_name)
        if not r.get("success", True):
            print(f"  XX  {label}: {r.get('error','?')}")
            continue
        ms = f"[{r['latency_ms']:.0f}ms]" if r.get("latency_ms") else ""
        print(f"\n  OK  {label}  {ms}")

        if task_name == "vad":
            print(f"      Speech: {r.get('speech_ratio',0)*100:.1f}%  "
                  f"| Duration: {r.get('total_speech_sec',0):.2f}s")
        elif task_name == "asr":
            print(f"      >> \"{r.get('text','')[:120]}\"")
            print(f"      Language: {r.get('language','N/A')}")
        elif task_name == "keyword_spotting":
            print(f"      Keyword: {r.get('top_label','')} "
                  f"({r.get('top_score',0)*100:.1f}%)")
        elif task_name == "speaker_id":
            print(f"      Speakers: {r.get('num_speakers',0)}")
        elif task_name == "emotion":
            print(f"      Emotion: {r.get('top_emotion','')} "
                  f"({r.get('top_score',0)*100:.1f}%)")
        elif task_name == "speech_quality":
            print(f"      DNSMOS: {r.get('dnsmos_overall',0):.2f}/5  "
                  f"| SNR: {r.get('snr_db',0):.1f}dB")
        elif task_name == "lang_accent_id":
            print(f"      Language: {r.get('top_language','')}  "
                  f"| Accent: {r.get('top_accent','N/A')}")
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
                  f"| Music: {r.get('music_fraction',0)*100:.1f}%")
        elif task_name == "anomaly_detection":
            flag = "** ANOMALY DETECTED **" if r.get("is_anomaly") else "Normal"
            print(f"      Status: {flag}  "
                  f"| Score: {r.get('anomaly_score',0):.4f}")

    print("\n" + "="*55)
    print("  Analysis complete. Listening for wake word again...")
    print("="*55 + "\n")


# ── Main loop ─────────────────────────────────────────────────────────────────

def listen_loop(wake_word: str, tasks: list, device: int, analysis_sec: int):
    """Continuously listen for wake word, run analysis when triggered."""
    import pyaudio
    pa, stream = _open_stream(device_index=device)

    print("="*55)
    print("  Edge Audio Framework — Wake Word Listener")
    print("="*55)
    print(f"  Wake word : \"{wake_word}\"")
    print(f"  Say       : \"Hi {wake_word}\" or \"{wake_word} start\"")
    print(f"  Tasks     : {', '.join(tasks)}")
    print(f"  Analysis  : {analysis_sec}s after trigger")
    print(f"  Device    : {'default' if device is None else device}")
    print("="*55)
    print("\nListening... (Ctrl+C to stop)\n")

    # Buffer for overlapping detection window
    prev_audio = np.zeros(0, dtype=np.float32)

    try:
        while True:
            # Record a short chunk for wake word detection
            chunk_audio = _record_seconds(stream, WAKE_CHUNK_SEC)

            # Combine with tail of previous chunk for overlap
            combined = np.concatenate([prev_audio, chunk_audio])
            prev_audio = chunk_audio[-int(OVERLAP_SAMPLES):]  # keep tail

            # Quick energy check — skip silent audio
            rms = float(np.sqrt(np.mean(combined**2)))
            if rms < 0.002:
                print(".", end="", flush=True)  # silence indicator
                continue

            print(f"\n[listening] RMS={rms:.4f} — checking for \"{wake_word}\"...")

            triggered, transcript = contains_wake_word(combined, wake_word)
            print(f"  Heard: \"{transcript}\"")

            if triggered:
                print(f"\n  *** WAKE WORD \"{wake_word}\" DETECTED! ***")
                print(f"  Recording {analysis_sec}s for full analysis...")
                print("  3... 2... 1...")

                # Record full analysis audio
                analysis_audio = _record_seconds(stream, analysis_sec)

                # Run analysis in same thread (pipeline is not thread-safe)
                run_full_analysis(analysis_audio, tasks)

                # Reset overlap buffer after analysis
                prev_audio = np.zeros(0, dtype=np.float32)
                print("Listening... (Ctrl+C to stop)\n")

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global OVERLAP_SAMPLES
    parser = argparse.ArgumentParser(
        description="Wake word listener — say 'Hi LTTS' to activate")
    parser.add_argument("--wake-word", "-w", default="LTTS",
                        help="Wake word/phrase to listen for (default: LTTS)")
    parser.add_argument("--tasks", "-t",
                        default=",".join(DEFAULT_TASKS),
                        help="Tasks to run after activation (comma-separated or 'all')")
    parser.add_argument("--duration", "-d", type=int, default=ANALYSIS_SEC,
                        help=f"Seconds to record for analysis (default: {ANALYSIS_SEC})")
    parser.add_argument("--device", type=int, default=None,
                        help="Mic device index (run with --list-devices to see options)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List microphone devices and exit")
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

    task_list = (ALL_TASKS if args.tasks.lower() == "all"
                 else [t.strip() for t in args.tasks.split(",")])

    OVERLAP_SAMPLES = int(WAKE_OVERLAP_SEC * SAMPLE_RATE)

    listen_loop(
        wake_word=args.wake_word.upper(),
        tasks=task_list,
        device=args.device,
        analysis_sec=args.duration,
    )


if __name__ == "__main__":
    main()
