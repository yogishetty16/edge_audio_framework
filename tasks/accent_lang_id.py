"""
tasks/accent_lang_id.py
=======================
Task 7: Language & Accent Identification.
Models:
  - Language ID: speechbrain/lang-id-voxlingua107-ecapa (107 languages, 80MB)
  - Accent ID:   Derived from Language ID confidence spread (no separate model needed)
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from core.audio_io import AudioData
from core.model_registry import registry, DEVICE, MODELS_DIR, get_hf_cache_dir
from core.result_schema import LanguageIDResult

logger = logging.getLogger(__name__)

LANG_MODEL_ID = "speechbrain/lang-id-voxlingua107-ecapa"

# English accent variants the VoxLingua107 model can distinguish
ENGLISH_ACCENT_MAP = {
    "en": "General English",
    "cy": "Welsh English",
    "ga": "Irish English",
    "gd": "Scottish English",
    "af": "South African English",
}


def _load_lang_model():
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    logger.info(f"Loading language ID model: {LANG_MODEL_ID}")
    save_dir = str(MODELS_DIR / "speechbrain_langid")
    classifier = EncoderClassifier.from_hparams(
        source=LANG_MODEL_ID,
        savedir=save_dir,
        run_opts={"device": DEVICE},
    )
    return classifier


registry.register("lang_id", _load_lang_model)


def _classify(waveform: np.ndarray, classifier, top_k: int = 5) -> Dict[str, float]:
    import torch
    signal = torch.from_numpy(waveform).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        # Run forward pass manually to avoid label_encoder.decode_torch KeyError
        embeddings = classifier.encode_batch(signal)
        out_prob = classifier.mods.classifier(embeddings).squeeze(1)

    probs = torch.nn.functional.softmax(out_prob, dim=-1).squeeze(0).tolist()
    if not isinstance(probs, list):
        probs = [probs]

    labels = classifier.hparams.label_encoder.ind2lab
    all_scores = {}
    for i in range(len(probs)):
        label = labels.get(i, f"lang_{i}")
        all_scores[label] = round(probs[i], 4)
    top = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    return dict(top[:top_k])


def analyze(audio: AudioData, **kwargs) -> LanguageIDResult:
    """
    Identify language and accent from audio using SpeechBrain ECAPA model.
    Returns LanguageIDResult with top language, accent, and all scores.
    """
    lang_scores: Dict[str, float] = {}
    accent_scores: Dict[str, float] = {}
    top_lang, top_lang_score = "", 0.0
    top_accent, top_accent_score = None, 0.0

    # Language ID
    try:
        classifier = registry.get("lang_id")
        raw_scores = _classify(audio.waveform, classifier, top_k=10)
        for k, v in raw_scores.items():
            # Extract base language code from formatting like "en: English"
            lang_code = k.split(":")[0].strip() if ":" in k else k.strip()
            lang_scores[lang_code] = v

        if lang_scores:
            top_lang, top_lang_score = max(lang_scores.items(), key=lambda x: x[1])
    except Exception as e:
        logger.error(f"Language ID failed: {e}", exc_info=True)

    # Accent ID — derived from language model's English-adjacent scores
    try:
        if top_lang in ("en", "cy", "ga", "gd", "af"):
            top_accent = ENGLISH_ACCENT_MAP.get(top_lang, "General English")
            top_accent_score = top_lang_score
            accent_scores = {
                ENGLISH_ACCENT_MAP[k]: lang_scores.get(k, 0.0)
                for k in ENGLISH_ACCENT_MAP if k in lang_scores
            }
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
