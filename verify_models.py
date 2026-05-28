#!/usr/bin/env python3
"""
verify_models.py
================
Dry-run diagnostic script to pre-load all deep learning models,
verify their configurations, and perform end-to-end execution checks
to ensure zero errors during a live demo.
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Enforce offline execution mode globally
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Force relative imports resolution
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

def repair_cache():
    print("[0/3] Checking and repairing Hugging Face cache snapshots...")
    from core.model_registry import MODELS_DIR
    import shutil
    
    repairs = {
        "models--superb--hubert-base-superb-er": {
            "snapshot_hash": "bac0e14e92f7f9fd56671c5060e572e883cad667",
            "files": [
                ("cefbc1394f3d04a79ceecc9eb31c9fec4511b168", ".gitattributes"),
                ("73bf2a90ec991ae6e5f171106f4e111837b0cb8e", "config.json"),
                ("d57e7d86d36a4a3d37fa0369fef294d00604550e", "preprocessor_config.json"),
                ("ed1f14f5e69bd7f4b0151286969c2eff4646505f", "README.md"),
                ("9e10f7a1672ad8f262012d3698bfbcdc232fe9483f9fbee3d9a1de405d134a2a", "pytorch_model.bin"),
            ]
        },
        "models--speechbrain--lang-id-voxlingua107-ecapa": {
            "snapshot_hash": "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9",
            "files": [
                ("feb65239285b042cfef21ce9af0e5fb376c075b2", ".gitattributes"),
                ("6fad8e826fd6808012326b33e4beb8d4e083d808", "config.json"),
                ("27d80047d277c45575779937a92cb829d4158528", "hyperparams.yaml"),
                ("a40a50dc26b99cbc5c0e44d3155755ff5c261b35", "README.md"),
                ("a50d9024ff58d317031c9787d4c6c614d454a87a8ef32f9d36338cd3ff57adbc", "classifier.ckpt"),
                ("ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a", "embedding_model.ckpt"),
                ("addb319892122ba2e7ddb4d01e0c87c4833dab38", "label_encoder.txt"),
                ("addb319892122ba2e7ddb4d01e0c87c4833dab38", "label_encoder.ckpt"),
                ("c369e01dfa2e0d84c6b116f33c7b94f1fe28c061642086538e93cde3d97c26ef", "normalizer.ckpt"),
                ("5bdc3a0a686eed58c6ccff264c5605a3c245e6598610dcc6e355104758acc6d7", "udhr_th.wav"),
            ]
        }
    }
    
    hf_cache_dir = MODELS_DIR / "hf_cache"
    for repo, data in repairs.items():
        repo_dir = hf_cache_dir / repo
        if not repo_dir.exists():
            print(f"  [WARN] Repository cache directory not found: {repo}")
            continue
            
        blobs_dir = repo_dir / "blobs"
        snapshot_dir = repo_dir / "snapshots" / data["snapshot_hash"]
        
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        for blob_hash, filename in data["files"]:
            blob_path = blobs_dir / blob_hash
            dest_path = snapshot_dir / filename
            
            if not blob_path.exists():
                print(f"  [WARN] Blob {blob_hash} missing for {repo} - {filename}")
                continue
                
            if not dest_path.exists() or dest_path.stat().st_size == 0:
                print(f"  [REPAIR] Copying {filename} to snapshot...")
                try:
                    shutil.copy2(str(blob_path), str(dest_path))
                except Exception as copy_err:
                    print(f"  [ERROR] Failed to copy {filename}: {copy_err}")
            else:
                pass
    print("  [OK] Cache check completed.")
    print()

def main():
    print("==================================================")
    print("    Edge Audio Framework — Model Verification     ")
    print("==================================================")
    print()

    repair_cache()

    task_modules = {
        "vad": "tasks.vad", 
        "asr": "tasks.asr",
        "keyword_spotting": "tasks.keyword_spotting",
        "speaker_id": "tasks.speaker_id", 
        "emotion": "tasks.emotion",
        "speech_quality": "tasks.speech_quality",
        "lang_accent_id": "tasks.accent_lang_id", 
        "esc": "tasks.esc",
        "acoustic_event_detection": "tasks.acoustic_event_detection",
        "audio_quality_monitoring": "tasks.audio_quality_monitor",
        "music_speech_detection": "tasks.music_speech_detection",
        "anomaly_detection": "tasks.anomaly_detection",
        "impulse_event": "tasks.impulse_event",
        "music_genre": "tasks.music_genre",
    }

    errors = []

    # ----------------------------------------------------------------------
    # [1/3] Importing Task Modules
    # ----------------------------------------------------------------------
    print("[1/3] Importing task modules & registering models...")
    import importlib
    loaded_mods = {}
    
    for task_name, mod_path in task_modules.items():
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(mod_path)
            loaded_mods[task_name] = mod
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [OK] {mod_path} imported in {elapsed:.1f}ms")
        except Exception as e:
            err_msg = f"Import error on {mod_path}: {e}"
            print(f"  [!] {err_msg}")
            errors.append(err_msg)
    print()

    # ----------------------------------------------------------------------
    # [2/3] Pre-loading Models from Registry
    # ----------------------------------------------------------------------
    print("[2/3] Pre-loading models into RAM...")
    try:
        from core.model_registry import registry
        registered_models = registry.list_registered()
        
        if not registered_models:
            print("  [WARN] No models registered in registry. Make sure tasks are imported.")
        else:
            for model_name in registered_models:
                t0 = time.perf_counter()
                try:
                    print(f"  Loading '{model_name}'...", end="", flush=True)
                    registry.get(model_name)
                    elapsed = time.perf_counter() - t0
                    print(f"\r  [OK] Model '{model_name}' loaded in {elapsed:.2f}s")
                except Exception as e:
                    print()
                    err_msg = f"Failed to load model '{model_name}': {e}"
                    print(f"  [!] {err_msg}")
                    traceback.print_exc()
                    errors.append(err_msg)
    except Exception as e:
        err_msg = f"Registry access failed: {e}"
        print(f"  [!] {err_msg}")
        errors.append(err_msg)
    print()

    # ----------------------------------------------------------------------
    # [3/3] Dry Run Audio Execution
    # ----------------------------------------------------------------------
    print("[3/3] Performing dry-run audio executions...")
    try:
        import numpy as np
        from core.audio_io import AudioData
        
        # Generate 5 seconds of mock float32 audio waveform at 16kHz
        sr = 16000
        duration = 5.0
        waveform = np.zeros(int(sr * duration), dtype=np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr, duration_sec=duration, n_channels=1)
        
        for task_name, mod in loaded_mods.items():
            t0 = time.perf_counter()
            try:
                # Run the task's analysis function
                result = mod.analyze(audio)
                elapsed = (time.perf_counter() - t0) * 1000
                
                # Check for output formatting
                res_dict = result.__dict__ if hasattr(result, '__dict__') else {}
                success = res_dict.get("success", True)
                err_detail = res_dict.get("error", None)
                
                if not success:
                    err_msg = f"Task '{task_name}' returned failure result: {err_detail}"
                    print(f"  [!] {err_msg}")
                    errors.append(err_msg)
                else:
                    print(f"  [OK] {task_name} execution: success in {elapsed:.1f}ms")
            except Exception as e:
                err_msg = f"Task '{task_name}' execution crashed: {e}"
                print(f"  [!] {err_msg}")
                traceback.print_exc()
                errors.append(err_msg)
    except Exception as e:
        err_msg = f"Mock waveform initialization failed: {e}"
        print(f"  [!] {err_msg}")
        errors.append(err_msg)
    print()

    # ----------------------------------------------------------------------
    # Final Diagnostic Verdict
    # ----------------------------------------------------------------------
    print("==================================================")
    if errors:
        print("  [ERROR] Verification finished with errors!")
        print(f"  Total errors detected: {len(errors)}")
        print("--------------------------------------------------")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        print("--------------------------------------------------")
        print("  Troubleshooting Recommendations:")
        print("  - Run 'python reassemble_models.py' if model files are missing.")
        print("  - Ensure PyAudio and virtual environment are fully active.")
        sys.exit(1)
    else:
        print("  [SUCCESS] All models loaded & executed cleanly!")
        print("            No errors found. Ready for demo run.")
    print("==================================================")

if __name__ == "__main__":
    main()
