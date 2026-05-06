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


def get_hf_cache_dir() -> str:
    """Return HuggingFace cache directory (respects HF_HOME env var)."""
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        return hf_home
    return str(MODELS_DIR / "hf_cache")


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
