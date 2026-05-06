"""
tasks/accent_lang_id.py
=======================
Task 7: Accent & Language Identification.
Models:
  - Language ID: facebook/mms-lid-126  (supports 126 languages)
  - Accent ID:   Jzuluaga/accent-id-commonaccent_xlsr-en-english (16 English accents)
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from core.audio_io import AudioData
from core.model_registry import registry, DEVICE, get_hf_cache_dir
from core.result_schema import LanguageIDResult

try:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
except ImportError:
    pass

logger = logging.getLogger(__name__)

LANG_MODEL_ID = "facebook/mms-lid-126"
ACCENT_MODEL_ID = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"


def _load_lang_model():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import torch

    logger.info(f"Loading language ID model: {LANG_MODEL_ID}")
    extractor = AutoFeatureExtractor.from_pretrained(LANG_MODEL_ID, cache_dir=get_hf_cache_dir())
    model = AutoModelForAudioClassification.from_pretrained(
        LANG_MODEL_ID, cache_dir=get_hf_cache_dir()
    ).to(DEVICE)
    model.eval()
    return {"model": model, "extractor": extractor}


def _load_accent_model():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import torch

    logger.info(f"Loading accent ID model: {ACCENT_MODEL_ID}")
    extractor = AutoFeatureExtractor.from_pretrained(
        ACCENT_MODEL_ID, cache_dir=get_hf_cache_dir()
    )
    model = AutoModelForAudioClassification.from_pretrained(
        ACCENT_MODEL_ID, cache_dir=get_hf_cache_dir()
    ).to(DEVICE)
    model.eval()
    return {"model": model, "extractor": extractor}


registry.register("lang_id", _load_lang_model)
registry.register("accent_id", _load_accent_model)


def _classify(waveform: np.ndarray, sr: int, bundle: dict, top_k: int = 5) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F

    model = bundle["model"]
    extractor = bundle["extractor"]

    max_len = 10 * sr
    waveform = waveform[:max_len]

    inputs = extractor(waveform, sampling_rate=sr, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze(0)
    id2label = model.config.id2label
    all_scores = {id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))}
    top = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    return dict(top[:top_k])


def analyze(audio: AudioData, **kwargs) -> LanguageIDResult:
    """
    Identify language and accent from audio.
    Returns LanguageIDResult with top language, accent, and all scores.
    """
    lang_scores: Dict[str, float] = {}
    accent_scores: Dict[str, float] = {}
    top_lang, top_lang_score = "", 0.0
    top_accent, top_accent_score = None, 0.0

    # Language ID
    try:
        bundle = registry.get("lang_id")
        lang_scores = _classify(audio.waveform, audio.sample_rate, bundle, top_k=10)
        if lang_scores:
            top_lang, top_lang_score = max(lang_scores.items(), key=lambda x: x[1])
    except Exception as e:
        logger.error(f"Language ID failed: {e}", exc_info=True)

    # Accent ID (English focused)
    try:
        bundle = registry.get("accent_id")
        accent_scores = _classify(audio.waveform, audio.sample_rate, bundle, top_k=5)
        if accent_scores:
            top_accent, top_accent_score = max(accent_scores.items(), key=lambda x: x[1])
    except Exception as e:
        logger.warning(f"Accent ID failed (non-fatal): {e}")

    return LanguageIDResult(
        top_language=top_lang,
        top_language_score=round(top_lang_score, 4),
        all_language_scores=lang_scores,
        top_accent=top_accent,
        top_accent_score=round(top_accent_score, 4),
        all_accent_scores=accent_scores,
    )
