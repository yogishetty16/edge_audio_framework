"""
core/model_registry.py
======================
Thread-safe lazy model registry.
Models are loaded on first use and cached in memory.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_FRAMEWORK_ROOT = Path(__file__).parent.parent          # edge_audio_framework/
MODELS_DIR: Path = _FRAMEWORK_ROOT / "models"           # edge_audio_framework/models/

# Global Torchaudio Patch for compatibility with newer versions
try:
    import torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None
except ImportError:
    pass

# Global Symlink Patch to prevent WinError 1314 on restrictive Windows environments
def _patch_symlink():
    import os, shutil, pathlib
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

def get_hf_cache_dir() -> str:
    """Return HuggingFace cache directory and globally enforce it for all backend libraries."""
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        cache_dir = hf_home
    else:
        cache_dir = str(MODELS_DIR / "hf_cache")
    
    # Force huggingface_hub to respect this directory globally
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    return cache_dir

# Initialize global cache enforcement immediately on import
get_hf_cache_dir()


# ── Device ─────────────────────────────────────────────────────────────────────
try:
    import torch
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

logger.debug(f"[model_registry] DEVICE={DEVICE}  MODELS_DIR={MODELS_DIR}")


# ── Registry ───────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Thread-safe lazy model loader.

    Usage:
        registry.register("my_model", lambda: load_my_model())
        model = registry.get("my_model")   # loads once, cached forever
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()

    def register(self, name: str, factory: Callable) -> None:
        """Register a factory function for a named model."""
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """Return the model instance, loading it lazily on first call."""
        if name in self._instances:
            return self._instances[name]

        # Ensure a per-model lock exists
        with self._meta_lock:
            if name not in self._model_locks:
                self._model_locks[name] = threading.Lock()

        with self._model_locks[name]:
            # Double-check inside lock
            if name not in self._instances:
                if name not in self._factories:
                    raise KeyError(
                        f"No factory registered for model '{name}'. "
                        f"Available: {list(self._factories.keys())}"
                    )
                logger.info(f"[registry] Loading '{name}'…")
                self._instances[name] = self._factories[name]()
                logger.info(f"[registry] '{name}' ready.")

        return self._instances[name]

    def is_loaded(self, name: str) -> bool:
        return name in self._instances

    def unload(self, name: str) -> None:
        """Remove a cached model from memory."""
        self._instances.pop(name, None)

    def list_registered(self):
        return list(self._factories.keys())


# Singleton
registry = ModelRegistry()
