"""
tasks/emotion.py
================
Task 4: Emotion / Sentiment Detection from speech.
Model: local HuBERT emotion model when available, with HF fallback.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from core.audio_io import AudioData
from core.result_schema import EmotionResult

# Lazy imports so SSL failures don't break the entire module at load time
try:
    from core.model_registry import registry, DEVICE, MODELS_DIR, get_hf_cache_dir
    _REGISTRY_OK = True
except Exception as _reg_err:
    _REGISTRY_OK = False
    import logging as _log
    _log.getLogger(__name__).warning(f"model_registry import failed: {_reg_err}")

try:
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, HubertForSequenceClassification
except ImportError:
    pass

logger = logging.getLogger(__name__)

AUDIO_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
LOCAL_AUDIO_MODEL_DIR = MODELS_DIR / "emotion_recognition" if _REGISTRY_OK else None
TEXT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

SENTIMENT_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

EMOTION_LABEL_MAP = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
    "fear": "fearful",
}


def _load_emotion_model():
    if LOCAL_AUDIO_MODEL_DIR and (LOCAL_AUDIO_MODEL_DIR / "pytorch_model.bin").exists():
        logger.info(f"Loading local HuBERT emotion model: {LOCAL_AUDIO_MODEL_DIR}")
        extractor = AutoFeatureExtractor.from_pretrained(
            str(LOCAL_AUDIO_MODEL_DIR),
            local_files_only=True,
        )
        model = HubertForSequenceClassification.from_pretrained(
            str(LOCAL_AUDIO_MODEL_DIR),
            local_files_only=True,
        )
    else:
        logger.info(f"Loading fallback emotion model: {AUDIO_MODEL_ID}")
        extractor = AutoFeatureExtractor.from_pretrained(
            AUDIO_MODEL_ID,
            cache_dir=get_hf_cache_dir(),
        )
        model = AutoModelForAudioClassification.from_pretrained(
            AUDIO_MODEL_ID,
            cache_dir=get_hf_cache_dir(),
            low_cpu_mem_usage=False,
        )
    model = model.to(DEVICE)
    model.eval()
    return {"model": model, "extractor": extractor}


def _load_sentiment_model():
    """Text sentiment — disabled: requires torch>=2.6 (CVE-2025-32434).
    Returns None so audio emotion still works."""
    return None


if _REGISTRY_OK:
    registry.register("emotion_audio", _load_emotion_model)
    registry.register("emotion_text", _load_sentiment_model)


def analyze(audio: AudioData, transcript: Optional[str] = None, **kwargs) -> EmotionResult:
    """Detect emotion from audio. Returns EmotionResult."""
    import torch
    import torch.nn.functional as F

    if not _REGISTRY_OK:
        return EmotionResult(success=False,
                             error="model_registry unavailable (SSL/import error). Run ssl_fix.py first.")

    emotion_scores: Dict[str, float] = {}
    top_emotion = ""
    top_score = 0.0
    sentiment = None

    # ── Audio emotion ─────────────────────────────────────────────────────────
    try:
        bundle = registry.get("emotion_audio")
        model = bundle["model"]
        extractor = bundle["extractor"]

        waveform = audio.waveform
        max_len = 10 * audio.sample_rate
        if len(waveform) > max_len:
            waveform = waveform[:max_len]

        inputs = extractor(
            waveform,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = F.softmax(logits, dim=-1).squeeze(0)
        id2label = model.config.id2label
        emotion_scores = {
            _normalize_emotion_label(id2label[i]): round(float(probs[i]), 4)
            for i in range(len(probs))
        }

        top_emotion, top_score = max(emotion_scores.items(), key=lambda x: x[1])
        top_score = round(top_score, 4)

    except Exception as e:
        logger.warning(f"Audio emotion detection unavailable: {e}")

    # ── Text sentiment (runs only if transcript provided) ─────────────────────
    if transcript and transcript.strip():
        try:
            pipe = registry.get("emotion_text")
            if pipe is not None:  # disabled when torch < 2.6
                result = pipe(transcript[:512])[0]
                raw_label = result["label"]
                sentiment = SENTIMENT_MAP.get(raw_label, raw_label.lower())
        except Exception as e:
            logger.warning(f"Text sentiment failed: {e}")


    if not top_emotion and not sentiment:
        return EmotionResult(success=False, error="Emotion detection produced no results")

    return EmotionResult(
        top_emotion=top_emotion,
        top_score=top_score,
        all_scores=emotion_scores,
        sentiment=sentiment,
    )


def _normalize_emotion_label(label: str) -> str:
    label = str(label or "").lower()
    return EMOTION_LABEL_MAP.get(label, label)
