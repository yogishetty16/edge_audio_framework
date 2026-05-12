"""
setup_manual_models.py
======================
After manually downloading model files from HuggingFace website,
run this script to configure the framework to use local model paths.

This creates a local_models.json config that tells each task to load
from the manual download folders instead of trying to download.
"""

import os
import sys
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
MODELS = ROOT / "models"
MANUAL = MODELS / "manual"
HF_CACHE = MODELS / "hf_cache"

def setup_hf_model(manual_name, hf_org, hf_model):
    """Move manually downloaded files into HuggingFace cache structure."""
    src = MANUAL / manual_name
    if not src.exists():
        print(f"  [SKIP] {manual_name} -- folder not found at {src}")
        return False

    files = list(src.iterdir())
    if not files:
        print(f"  [SKIP] {manual_name} -- folder is empty")
        return False

    # Create HF cache structure:
    # models/hf_cache/models--org--model/snapshots/manual/
    cache_model = HF_CACHE / f"models--{hf_org}--{hf_model}"
    snapshot_dir = cache_model / "snapshots" / "manual"
    refs_dir = cache_model / "refs"

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    # Write refs/main pointing to our "manual" snapshot
    (refs_dir / "main").write_text("manual")

    # Copy all files to the snapshot directory
    for f in files:
        if f.is_file():
            dest = snapshot_dir / f.name
            if not dest.exists():
                shutil.copy2(str(f), str(dest))
                print(f"    Copied: {f.name}")
            else:
                print(f"    Exists: {f.name}")

    print(f"  [OK] {hf_org}/{hf_model}")
    return True


def setup_whisper(manual_name):
    """Move Whisper files into the faster-whisper directory."""
    src = MANUAL / manual_name
    if not src.exists():
        print(f"  [SKIP] {manual_name} -- folder not found")
        return False

    dest = MODELS / "faster-whisper-base"
    dest.mkdir(parents=True, exist_ok=True)

    for f in src.iterdir():
        if f.is_file():
            target = dest / f.name
            if not target.exists():
                shutil.copy2(str(f), str(target))
                print(f"    Copied: {f.name}")
            else:
                print(f"    Exists: {f.name}")

    print(f"  [OK] faster-whisper-base")
    return True


def check_speechbrain(folder_name, model_name):
    """Verify SpeechBrain model files are in place."""
    path = MODELS / folder_name
    if not path.exists():
        print(f"  [SKIP] {model_name} -- folder not found at {path}")
        return False

    required = ["hyperparams.yaml"]
    missing = [f for f in required if not (path / f).exists()]
    if missing:
        print(f"  [SKIP] {model_name} -- missing files: {', '.join(missing)}")
        return False

    print(f"  [OK] {model_name}")
    return True


def check_silero():
    """Check if Silero VAD is available."""
    home = Path.home()
    torch_hub = home / ".cache" / "torch" / "hub"
    silero_dirs = list(torch_hub.glob("snakers4_silero-vad*")) if torch_hub.exists() else []

    if silero_dirs:
        print(f"  [OK] silero-vad (found at {silero_dirs[0]})")
        return True

    # Also check local torch hub
    local_hub = MODELS / "torch_hub"
    if local_hub.exists() and list(local_hub.glob("snakers4*")):
        print(f"  [OK] silero-vad (found at {local_hub})")
        return True

    print("  [SKIP] silero-vad -- not found")
    print("         Download from: https://github.com/snakers4/silero-vad")
    print("         Extract to: ~/.cache/torch/hub/snakers4_silero-vad_master/")
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("  Manual Model Setup")
    print("  Organizing downloaded files into framework structure")
    print("=" * 60)

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    results = {}

    print("\n[1/8] Whisper base (ASR)...")
    results["whisper"] = setup_whisper("whisper-base")

    print("\n[2/8] Language ID (SpeechBrain VoxLingua107)...")
    results["lang_id"] = check_speechbrain("speechbrain_langid", "lang-id-voxlingua107")

    print("\n[3/8] Emotion (SUPERB HuBERT-base)...")
    results["emotion"] = setup_hf_model("hubert-emotion", "superb", "hubert-base-superb-er")

    print("\n[4/8] Environmental Sound (MIT AST AudioSet)...")
    results["esc"] = setup_hf_model("ast-audioset", "MIT", "ast-finetuned-audioset-10-10-0.4593")

    print("\n[5/8] Keyword Spotting (MIT AST Speech Commands)...")
    results["keyword"] = setup_hf_model("ast-speech-commands", "MIT", "ast-finetuned-speech-commands-v2")

    print("\n[6/8] Music Genre (DistilHuBERT GTZAN)...")
    results["genre"] = setup_hf_model("distilhubert-gtzan", "sanchit-gandhi", "distilhubert-finetuned-gtzan")

    print("\n[7/8] Speaker ID (SpeechBrain ECAPA)...")
    results["speaker"] = check_speechbrain("speechbrain_ecapa", "spkrec-ecapa-voxceleb")

    print("\n[8/8] Silero VAD...")
    results["vad"] = check_silero()

    # Summary
    ok = sum(1 for v in results.values() if v)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"  Setup complete: {ok}/{total} models ready")
    if ok < total:
        missing = [k for k, v in results.items() if not v]
        print(f"  Missing: {', '.join(missing)}")
        print("  Download the missing model files and run this script again.")
    else:
        print("  All models are in place. You can now run:")
        print("    python fast_run.py --agent --tasks vad,asr,emotion")
    print("=" * 60)
