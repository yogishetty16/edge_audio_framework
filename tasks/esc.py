"""
tasks/esc.py  [EDGE EDITION]
============================
Task 8: Environmental Sound Classification.
Model: MIT/ast-finetuned-audioset-10-10-0.4593 (Audio Spectrogram Transformer)
ONNX INT8 quantized — ~90MB, ~10ms inference on CPU.
Classifies into 527 AudioSet classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.audio_io import AudioData
from core.model_registry import registry, DEVICE, MODELS_DIR, get_hf_cache_dir
from core.result_schema import ESCResult

try:
    from transformers import AutoFeatureExtractor, ASTForAudioClassification, AutoConfig
except ImportError:
    pass

logger = logging.getLogger(__name__)

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
ONNX_DIR = MODELS_DIR / "MIT_ast-finetuned-audioset-10-10-0.4593_int8"

# Friendly top-level ESC-50 labels mapped to AudioSet classes for display
TOP_LABELS_DISPLAY = 5


def _load_model():
    """Try ONNX INT8 first; fall back to HuggingFace PyTorch."""
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    onnx_path = ONNX_DIR / "model_quantized.onnx"
    if onnx_path.exists():
        logger.info(f"Loading AST ONNX INT8: {onnx_path}")
        from core.onnx_utils import create_onnx_session
        session = create_onnx_session(str(onnx_path))
        extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir=get_hf_cache_dir())
        id2label = _load_id2label()
        return {"type": "onnx", "session": session, "extractor": extractor, "id2label": id2label}

    # Fallback: HuggingFace PyTorch
    logger.info(f"Loading AST (PyTorch): {MODEL_ID}")
    import torch

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir=get_hf_cache_dir())
    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID, cache_dir=get_hf_cache_dir(), device_map=DEVICE
    )
    model.eval()
    id2label = model.config.id2label
    return {"type": "torch", "model": model, "extractor": extractor, "id2label": id2label}


def _load_id2label() -> dict:
    """Load id2label from config.json in HF cache."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_ID, cache_dir=get_hf_cache_dir())
        return cfg.id2label
    except Exception:
        return {}


registry.register("esc", _load_model)


def analyze(
    audio: AudioData,
    labels: Optional[List[str]] = None,
    top_k: int = 5,
    **kwargs,
) -> ESCResult:
    """
    Classify environmental sounds into AudioSet-527 categories.
    Returns ESCResult with top class and confidence scores.
    """
    try:
        bundle = registry.get("esc")
        extractor = bundle["extractor"]
        id2label = bundle["id2label"]

        # AST expects 16kHz, max 10s
        waveform = audio.waveform[:10 * audio.sample_rate]

        inputs = extractor(
            waveform,
            sampling_rate=audio.sample_rate,
            return_tensors="pt" if bundle["type"] == "torch" else "np",
        )

        if bundle["type"] == "onnx":
            session = bundle["session"]
            inp_name = session.get_inputs()[0].name
            logits = session.run(None, {inp_name: inputs["input_values"].astype(np.float32)})[0]
            probs = _softmax(logits[0])
        else:
            import torch
            import torch.nn.functional as F
            with torch.no_grad():
                logits = bundle["model"](**{k: v.to(DEVICE) for k, v in inputs.items()}).logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        if id2label:
            all_scores: Dict[str, float] = {
                id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))
            }
        else:
            all_scores = {f"class_{i}": round(float(probs[i]), 4) for i in range(len(probs))}

        # If user specified a whitelist, filter
        if labels:
            all_scores = {k: v for k, v in all_scores.items()
                         if any(l.lower() in k.lower() for l in labels)}

        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        top_class = sorted_scores[0][0] if sorted_scores else ""
        top_score = sorted_scores[0][1] if sorted_scores else 0.0

        return ESCResult(
            top_class=top_class,
            top_score=round(top_score, 4),
            all_scores=dict(sorted_scores[:top_k]),
        )

    except Exception as e:
        logger.error(f"ESC failed: {e}", exc_info=True)
        return ESCResult(success=False, error=str(e))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
