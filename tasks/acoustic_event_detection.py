"""
tasks/acoustic_event_detection.py  [EDGE EDITION]
==================================================
Task 9: Acoustic Event Detection.
Model: PANNs MobileNetV2 ONNX (14MB, 527 AudioSet classes, ~8ms inference)
Fast enough for real-time edge deployment.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

from core.audio_io import AudioData
from core.model_registry import registry, DEVICE, MODELS_DIR, get_hf_cache_dir
from core.result_schema import AEDResult, AcousticEvent

logger = logging.getLogger(__name__)

PANNS_REPO = MODELS_DIR / "audioset_tagging_cnn"
PANNS_ONNX_PATH = MODELS_DIR / "panns_mobilenetv2.onnx"
PANNS_SR = 32_000

# Fallback if PANNs not available: use AST from HuggingFace
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


def _load_model():
    """Load PANNs MobileNetV2 ONNX if available, else AST from HF."""
    # 1) Try local ONNX
    if PANNS_ONNX_PATH.exists():
        from core.onnx_utils import create_onnx_session
        session = create_onnx_session(str(PANNS_ONNX_PATH))
        labels = _load_audioset_labels()
        return {"type": "panns_onnx", "session": session, "labels": labels}

    # 2) Try PANNs PyTorch from local clone
    pytorch_path = PANNS_REPO / "pytorch"
    if pytorch_path.exists():
        try:
            return _load_panns_torch()
        except Exception as e:
            logger.info(f"PANNs PyTorch unavailable, falling back to AST: {e}")

    # 3) Fallback to AST (same model as ESC task — shared registry entry)
    logger.info(f"PANNs not found, using AST for AED from HF cache: {AST_MODEL_ID}")
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    import torch
    try:
        extractor = AutoFeatureExtractor.from_pretrained(
            AST_MODEL_ID, cache_dir=get_hf_cache_dir(), local_files_only=True,
        )
        model = ASTForAudioClassification.from_pretrained(
            AST_MODEL_ID, cache_dir=get_hf_cache_dir(), local_files_only=True,
        )
    except Exception as e:
        logger.error(
            f"Model not found at {get_hf_cache_dir()}/models--{AST_MODEL_ID.replace('/', '--')} "
            f"-- download with: python download_models.py --task acoustic_event_detection. Error: {e}"
        )
        raise
    model.eval().to(DEVICE)
    return {"type": "ast", "model": model, "extractor": extractor,
            "id2label": model.config.id2label}


def _load_panns_torch():
    import torch
    if str(PANNS_REPO / "pytorch") not in sys.path:
        sys.path.insert(0, str(PANNS_REPO / "pytorch"))
    from models import MobileNetV2
    checkpoint_path = PANNS_REPO / "MobileNetV2_mAP=0.383.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PANNs checkpoint not found: {checkpoint_path}")
    model = MobileNetV2(sample_rate=PANNS_SR, window_size=1024,
                        hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                        classes_num=527)
    cp = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(cp["model"])
    model.eval().to(DEVICE)
    labels = _load_audioset_labels()
    return {"type": "panns_torch", "model": model, "labels": labels}


registry.register("aed", _load_model)


def analyze(audio: AudioData, top_k: int = 10, **kwargs) -> AEDResult:
    """Detect acoustic events. Returns top-k events with scores."""
    try:
        bundle = registry.get("aed")

        if bundle["type"] == "panns_onnx":
            return _run_panns_onnx(audio, bundle, top_k)
        elif bundle["type"] == "panns_torch":
            return _run_panns_torch(audio, bundle, top_k)
        elif bundle["type"] == "ast":
            return _run_ast(audio, bundle, top_k)
        return AEDResult(success=False, error="Unknown AED model type")

    except Exception as e:
        logger.error(f"AED failed: {e}", exc_info=True)
        return AEDResult(success=False, error=str(e))


def _resample_to_panns(audio: AudioData) -> np.ndarray:
    import librosa
    if audio.sample_rate == PANNS_SR:
        return audio.waveform
    return librosa.resample(audio.waveform, orig_sr=audio.sample_rate, target_sr=PANNS_SR)


def _run_panns_onnx(audio: AudioData, bundle: dict, top_k: int) -> AEDResult:
    wv = _resample_to_panns(audio).reshape(1, -1).astype(np.float32)
    session = bundle["session"]
    labels = bundle["labels"]
    inp_name = session.get_inputs()[0].name
    out = session.run(None, {inp_name: wv})
    probs = _softmax(out[0][0])
    return _build_result(probs, labels, audio.duration_sec, top_k)


def _run_panns_torch(audio: AudioData, bundle: dict, top_k: int) -> AEDResult:
    import torch
    wv = _resample_to_panns(audio)
    wv_t = torch.from_numpy(wv).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = bundle["model"](wv_t, None)
    probs = out["clipwise_output"].squeeze(0).cpu().numpy()
    return _build_result(probs, bundle["labels"], audio.duration_sec, top_k)


def _run_ast(audio: AudioData, bundle: dict, top_k: int) -> AEDResult:
    import torch, torch.nn.functional as F
    extractor = bundle["extractor"]
    model = bundle["model"]
    id2label = bundle["id2label"]
    inputs = extractor(audio.waveform[:10*audio.sample_rate],
                       sampling_rate=audio.sample_rate, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    labels = [id2label.get(i, f"class_{i}") for i in range(len(probs))]
    return _build_result(probs, labels, audio.duration_sec, top_k)


def _build_result(probs: np.ndarray, labels: list, duration: float, top_k: int) -> AEDResult:
    top_idx = np.argsort(probs)[::-1][:top_k]
    events = [
        AcousticEvent(
            label=labels[i] if i < len(labels) else f"class_{i}",
            start_sec=0.0,
            end_sec=duration,
            score=round(float(probs[i]), 4),
        )
        for i in top_idx
    ]
    return AEDResult(events=events, frame_labels=[e.label for e in events[:5]])


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def _load_audioset_labels() -> List[str]:
    labels_csv = PANNS_REPO / "metadata" / "class_labels_indices.csv"
    try:
        import csv
        with open(labels_csv) as f:
            reader = csv.reader(f)
            next(reader)
            return [row[2] for row in reader]
    except Exception:
        return [f"class_{i}" for i in range(527)]
