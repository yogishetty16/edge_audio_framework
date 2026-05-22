"""
download_all_models.py
======================
Run this script on an unrestricted network to download and cache
ALL required models for the Edge Audio Framework into the local 'models' folder.
"""

import os
from pathlib import Path
import logging

# Ensure logging shows us what is happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import our custom registry and configuration to enforce the local cache paths
from core.model_registry import registry


# All 13 tasks
TASKS_TO_DOWNLOAD = [
    "vad",
    "asr",
    "keyword_spotting",
    "speaker_id",
    "emotion",
    "speech_quality",
    "lang_accent_id",
    "esc",
    "acoustic_event_detection",
    "audio_quality_monitoring",
    "music_speech_detection",
    "anomaly_detection",
    "impulse_event"
]

def main():
    print("=" * 60)
    print("  Downloading ALL models to local cache...")
    print("  This may take several minutes depending on your connection.")
    print("=" * 60)
    
    # Import tasks to register them with the model registry
    from tasks import (
        vad, asr, keyword_spotting, speaker_id, emotion,
        speech_quality, accent_lang_id, esc, acoustic_event_detection,
        audio_quality_monitoring, music_speech_detection, anomaly_detection,
        impulse_event
    )
    
    success_count = 0
    fail_count = 0
    
    for task_name in TASKS_TO_DOWNLOAD:
        print(f"\n[DOWNLOADING] Task: {task_name}")
        try:
            # Getting the model from the registry forces it to download and cache
            _ = registry.get(task_name)
            print(f"  [OK] {task_name} downloaded successfully.")
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to download {task_name}:\n{e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"  DOWNLOAD COMPLETE")
    print(f"  Success: {success_count} | Failed: {fail_count}")
    print("=" * 60)
    print("\nIf all tasks succeeded, your 'models' folder is now fully populated!")
    print("You can zip the 'edge_audio_framework' folder and copy it to your offline Linux machine.")

if __name__ == "__main__":
    main()
