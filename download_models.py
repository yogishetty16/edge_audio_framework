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

# ==========================================
# 2. DOWNLOAD LATEST MODELS
# ==========================================
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel

print("=============================================================")
print("  🚀 Edge Framework - SOTA Model Downloader (Proxy Bypassed) ")
print("=============================================================")

print("\n[1/9] Downloading Whisper tiny.en (ASR)...")
WhisperModel("tiny.en", device="auto", compute_type="int8", download_root=os.environ["HF_HOME"])

print("\n[2/9] Downloading Language ID (Facebook MMS)...")
snapshot_download(repo_id="facebook/mms-lid-126", cache_dir=os.environ["HF_HOME"])

print("\n[3/9] Downloading Accent ID (Jzuluaga XLS-R)...")
snapshot_download(repo_id="Jzuluaga/accent-id-commonaccent_xlsr-en-english", cache_dir=os.environ["HF_HOME"])

print("\n[4/9] Downloading Emotion Recognition (ehcalabres Wav2Vec2)...")
snapshot_download(repo_id="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", cache_dir=os.environ["HF_HOME"])

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

print("\n✅ All Required Models downloaded directly into the proper cache folders!")
