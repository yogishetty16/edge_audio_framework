import os
import ssl
import sys
import torch

# ==========================================
# 1. CORPORATE NETWORK / SSL PROXY BYPASS
# ==========================================
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Force models to download into the correct cache folders
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(_ROOT, "models", "hf_cache")
os.environ["TORCH_HOME"] = os.path.join(_ROOT, "models", "torch_hub")

try:
    import urllib3; urllib3.disable_warnings()
except ImportError: pass

try:
    import requests
    _orig_req = requests.Session.request
    def _patched_req(self, *args, **kwargs):
        kwargs["verify"] = False
        return _orig_req(self, *args, **kwargs)
    requests.Session.request = _patched_req
except ImportError: pass

import shutil
import pathlib

def _patch_symlink():
    if hasattr(os, "symlink"):
        _orig_symlink = os.symlink
        def safe_symlink(src, dst, *args, **kwargs):
            try: _orig_symlink(src, dst, *args, **kwargs)
            except OSError:
                if os.path.exists(dst) or os.path.islink(dst):
                    try: os.remove(dst)
                    except: pass
                try:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.isdir(src): shutil.copytree(src, dst)
                    else: shutil.copyfile(src, dst)
                except Exception: pass
        os.symlink = safe_symlink

    if hasattr(pathlib.Path, "symlink_to"):
        _orig_pathlib_symlink = pathlib.Path.symlink_to
        def safe_pathlib_symlink(self, target, target_is_directory=False):
            try: _orig_pathlib_symlink(self, target, target_is_directory)
            except OSError:
                if self.exists() or self.is_symlink():
                    try: self.unlink()
                    except: pass
                try:
                    os.makedirs(str(self.parent), exist_ok=True)
                    t = str(target)
                    if os.path.isdir(t): shutil.copytree(t, str(self))
                    else: shutil.copyfile(t, str(self))
                except Exception: pass
        pathlib.Path.symlink_to = safe_pathlib_symlink

_patch_symlink()

# Patch torchaudio for SpeechBrain compatibility
import torchaudio
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None

# ==========================================
# 2. DOWNLOAD LATEST MODELS
# ==========================================
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel

print("=============================================================")
print("  Edge Framework - SOTA Model Downloader (Proxy Bypassed) ")
print("=============================================================")

print("\n[1/9] Downloading Whisper base (ASR)...")
WhisperModel("base", device="auto", compute_type="int8", download_root=os.environ["HF_HOME"])

print("\n[2/9] Downloading Language ID (SpeechBrain VoxLingua107)...")
from huggingface_hub import hf_hub_download
_lang_save = os.path.join(_ROOT, "models", "speechbrain_langid")
os.makedirs(_lang_save, exist_ok=True)
_lang_repo = "speechbrain/lang-id-voxlingua107-ecapa"
for _fname in ["hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt"]:
    hf_hub_download(repo_id=_lang_repo, filename=_fname, local_dir=_lang_save)
    print(f"  -> {_fname}")
print("  Language ID model cached.")

print("\n[3/9] Accent ID — derived from Language ID (no separate download needed)...")

print("\n[4/9] Downloading Emotion Recognition (SUPERB HuBERT-base)...")
snapshot_download(repo_id="superb/hubert-base-superb-er", cache_dir=os.environ["HF_HOME"])

print("\n[5/9] Downloading Environment Sound (MIT AST)...")
snapshot_download(repo_id="MIT/ast-finetuned-audioset-10-10-0.4593", cache_dir=os.environ["HF_HOME"])

print("\n[6/9] Downloading Keyword Spotting (MIT AST Speech Commands)...")
snapshot_download(repo_id="MIT/ast-finetuned-speech-commands-v2", cache_dir=os.environ["HF_HOME"])

print("\n[7/9] Downloading Music Genre (DistilHuBERT)...")
snapshot_download(repo_id="sanchit-gandhi/distilhubert-finetuned-gtzan", cache_dir=os.environ["HF_HOME"])

print("\n[8/9] Downloading Speaker Identification (VoxCeleb)...")
snapshot_download(repo_id="speechbrain/spkrec-ecapa-voxceleb", cache_dir=os.environ["HF_HOME"])

print("\n[9/9] Caching Silero VAD (Voice Activity)...")
torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, trust_repo=True)

print("\n[DONE] All 9 models downloaded into the local models/ folder!")
