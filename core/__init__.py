"""
Edge Audio Framework — core package
"""
from .audio_io import AudioData, load_audio, split_into_chunks
from .model_registry import registry, DEVICE, MODELS_DIR, get_hf_cache_dir
