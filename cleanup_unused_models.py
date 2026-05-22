"""
cleanup_unused_models.py
========================
Moves massive leftover models from the local cache to a backup folder 
outside the framework, ensuring the final zip file is as lightweight as possible.
"""

import os
import shutil
from pathlib import Path

# Paths that contain massive models we DO NOT USE
UNUSED_MODELS = [
    r"models\hf_cache\models--facebook--mms-lid-126",
    r"models\hf_cache\models--ehcalabres--wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    r"models\models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
    r"models\hf_cache\models--cardiffnlp--twitter-roberta-base-sentiment-latest",
    r"models\hf_cache\models--Jzuluaga--accent-id-commonaccent_xlsr-en-english",
    r"models\emotion_recognition",
    r"models\audio_classification",
]

BACKUP_DIR = Path(r"F:\unused_models_backup")

def main():
    print("=" * 60)
    print(f"  Moving massive unused models to {BACKUP_DIR} ...")
    print("=" * 60)
    
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    freed_space = 0
    
    for path_str in UNUSED_MODELS:
        path = Path(path_str)
        if path.exists():
            # Calculate size
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if path.is_dir() else path.stat().st_size
            
            try:
                dest = BACKUP_DIR / path.name
                shutil.move(str(path), str(dest))
                
                mb = size / (1024 * 1024)
                freed_space += size
                print(f"  [MOVED] {path.name} (Freed {mb:.1f} MB)")
            except Exception as e:
                print(f"  [ERROR] Could not move {path}: {e}")

    total_mb = freed_space / (1024 * 1024)
    print("=" * 60)
    print(f"  Move Complete! Safely backed up {total_mb:.1f} MB of space.")
    print("  Your framework folder is now fully lightweight and ready to zip!")
    print("=" * 60)

if __name__ == "__main__":
    main()
