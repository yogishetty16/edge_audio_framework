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

AUDIO_MODEL_ID = "superb/hubert-base-superb-er"
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
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "surprise": "surprise",
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
        logger.info(f"Loading emotion model from HF cache: {AUDIO_MODEL_ID}")
        try:
            extractor = AutoFeatureExtractor.from_pretrained(
                AUDIO_MODEL_ID,
                cache_dir=get_hf_cache_dir(),
                local_files_only=True,
            )
            model = AutoModelForAudioClassification.from_pretrained(
                AUDIO_MODEL_ID,
                cache_dir=get_hf_cache_dir(),
                local_files_only=True,
            )
        except Exception as e:
            logger.error(
                f"Model not found at {get_hf_cache_dir()}/models--{AUDIO_MODEL_ID.replace('/', '--')} "
                f"-- download with: python download_models.py --task emotion. Error: {e}"
            )
            raise
    model.eval().to(DEVICE)
    return {"model": model, "extractor": extractor}


def _load_sentiment_model():
    """Text sentiment — disabled: requires torch>=2.6 (CVE-2025-32434).
    Returns None so audio emotion still works."""
    return None


if _REGISTRY_OK:
    registry.register("emotion_audio", _load_emotion_model)
    registry.register("emotion_text", _load_sentiment_model)


from dataclasses import dataclass

@dataclass
class CustomEmotionResult(EmotionResult):
    low_confidence_note: Optional[str] = None
    skipped: Optional[bool] = None
    skip_reason: Optional[str] = None
    low_confidence: Optional[bool] = None

    def __setitem__(self, key, value):
        setattr(self, key, value)


def run(waveform, sr, vad_result=None):
    speech_ratio = 0.0
    if vad_result:
        if isinstance(vad_result, dict):
            speech_ratio = vad_result.get("speech_ratio", 0.0)
        else:
            speech_ratio = getattr(vad_result, "speech_ratio", 0.0)

    if speech_ratio < 0.25:
        return {
            "top_emotion": "neutral",
            "top_score": 0.0,
            "all_emotions": {},
            "success": True,
            "skipped": True,
            "skip_reason": "insufficient_speech",
            "speech_ratio": speech_ratio
        }

    if not _REGISTRY_OK:
        return {
            "top_emotion": "neutral",
            "top_score": 0.0,
            "all_emotions": {},
            "success": False,
            "error": "model_registry unavailable"
        }

    try:
        import torch
        import torch.nn.functional as F

        bundle = registry.get("emotion_audio")
        model = bundle["model"]
        extractor = bundle["extractor"]

        max_len = 10 * sr
        if len(waveform) > max_len:
            waveform = waveform[:max_len]

        inputs = extractor(
            waveform,
            sampling_rate=sr,
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

        # FIX 3 — Happy suppression on non-speech context
        if top_emotion == "happy" and top_score < 0.70:
            if speech_ratio < 0.50:
                non_happy_scores = {k: v for k, v in emotion_scores.items() if k != "happy"}
                if non_happy_scores:
                    alt = max(non_happy_scores, key=lambda k: non_happy_scores[k])
                    alt_score = non_happy_scores[alt]
                    if alt_score > 0.25:
                        top_emotion = alt
                        top_score = alt_score
                    else:
                        top_emotion = "neutral"
                        top_score = emotion_scores.get("neutral", 0.0)
                else:
                    top_emotion = "neutral"
                    top_score = emotion_scores.get("neutral", 0.0)

        # FIX 2 — Confidence threshold gate
        CONFIDENCE_THRESHOLD = 0.55
        if top_score < CONFIDENCE_THRESHOLD:
            return {
                "top_emotion": "neutral",
                "top_score": top_score,
                "all_emotions": emotion_scores,
                "success": True,
                "low_confidence": True,
                "low_confidence_note": (
                    f"Top emotion '{top_emotion}' at "
                    f"{top_score:.1%} below threshold "
                    f"{CONFIDENCE_THRESHOLD:.0%} — "
                    f"returning neutral"
                )
            }

        return {
            "top_emotion": top_emotion,
            "top_score": top_score,
            "all_emotions": emotion_scores,
            "success": True
        }

    except Exception as e:
        logger.warning(f"Audio emotion detection failed: {e}")
        return {
            "top_emotion": "neutral",
            "top_score": 0.0,
            "all_emotions": {},
            "success": False,
            "error": str(e)
        }


def analyze(audio: AudioData, transcript: Optional[str] = None, **kwargs) -> EmotionResult:
    """Detect emotion from audio. Returns EmotionResult."""
    vad_result = kwargs.get("vad_result")
    if vad_result is None:
        try:
            import tasks.vad as vad_task
            vad_result = vad_task.analyze(audio)
        except Exception:
            pass

    run_res = run(audio.waveform, audio.sample_rate, vad_result)

    success = run_res.get("success", True)
    error = run_res.get("error") if not success else None

    return CustomEmotionResult(
        success=success,
        error=error,
        top_emotion=run_res.get("top_emotion", ""),
        top_score=run_res.get("top_score", 0.0),
        all_scores=run_res.get("all_emotions", {}),
        sentiment=None,
        low_confidence_note=run_res.get("low_confidence_note"),
        skipped=run_res.get("skipped"),
        skip_reason=run_res.get("skip_reason"),
        low_confidence=run_res.get("low_confidence")
    )


def _normalize_emotion_label(label: str) -> str:
    label = str(label or "").lower()
    return EMOTION_LABEL_MAP.get(label, label)
