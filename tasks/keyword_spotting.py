"""
tasks/keyword_spotting.py
=========================
Task 2: Keyword Spotting using Audio Spectrogram Transformer (AST).
Model: MIT/ast-finetuned-speech-commands-v2
Detects 35 Google Speech Commands (yes, no, stop, go, up, down, etc.)
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from core.audio_io import AudioData
from core.model_registry import registry, DEVICE, get_hf_cache_dir
from core.result_schema import KeywordResult

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
except ImportError:
    pass

MODEL_ID = "MIT/ast-finetuned-speech-commands-v2"

# Configurable keyword whitelist — None means return top‑5 from all 35 commands
KEYWORD_WHITELIST = None   # e.g., ["yes", "no", "stop", "go"]


def _load_model():
    logger.info(f"Loading AST keyword model: {MODEL_ID}")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir=get_hf_cache_dir())
    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID,
        cache_dir=get_hf_cache_dir(),
        device_map=DEVICE
    )
    model.eval()
    return {"model": model, "extractor": extractor}


registry.register("keyword_spotting", _load_model)


def analyze(audio: AudioData, keywords: List[str] = None, top_k: int = 5, **kwargs) -> KeywordResult:
    """
    Classify audio into speech commands / keywords.
    Returns KeywordResult with detected keywords and confidence scores.
    """
    import torch
    import torch.nn.functional as F

    try:
        bundle = registry.get("keyword_spotting")
        model = bundle["model"]
        extractor = bundle["extractor"]

        # AST expects 16kHz and up to 10s audio
        waveform = audio.waveform
        sr = audio.sample_rate
        max_len = 10 * sr
        if len(waveform) > max_len:
            waveform = waveform[:max_len]

        inputs = extractor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = F.softmax(logits, dim=-1).squeeze(0)
        id2label = model.config.id2label

        # Build score dict
        all_scores: Dict[str, float] = {
            id2label[i]: float(probs[i]) for i in range(len(probs))
        }

        # Filter by whitelist if provided
        filter_list = keywords or KEYWORD_WHITELIST
        if filter_list:
            all_scores = {k: v for k, v in all_scores.items() if k.lower() in [kw.lower() for kw in filter_list]}

        # Sort and take top_k
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        top_scores = dict(sorted_scores[:top_k])

        detected = [k for k, v in top_scores.items() if v > 0.5]
        top_label = sorted_scores[0][0] if sorted_scores else ""
        top_score = sorted_scores[0][1] if sorted_scores else 0.0

        return KeywordResult(
            detected_keywords=detected,
            scores=top_scores,
            top_label=top_label,
            top_score=round(top_score, 4),
        )

    except Exception as e:
        logger.error(f"Keyword spotting failed: {e}", exc_info=True)
        return KeywordResult(success=False, error=str(e))
