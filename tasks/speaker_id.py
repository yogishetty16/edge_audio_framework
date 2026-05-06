"""
tasks/speaker_id.py  [EDGE EDITION]
=====================================
Task 3: Speaker Identification & Verification.
Model: SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb, ~20MB)
Compatible with torchaudio 2.x via backward-compat patch.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from core.audio_io import AudioData
from core.model_registry import registry, DEVICE, MODELS_DIR, get_hf_cache_dir
from core.result_schema import SpeakerIDResult, SpeakerSegment, SpeakerVerificationResult


def _patch_torchaudio():
    """Patch removed torchaudio APIs for speechbrain 0.5.x compatibility."""
    try:
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda b: None
        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"
    except ImportError:
        pass

logger = logging.getLogger(__name__)


# ─── ECAPA-TDNN Speaker Embeddings ───────────────────────────────────────────

def _load_ecapa():
    """Load SpeechBrain ECAPA-TDNN for speaker embeddings (with torchaudio 2.x compat patch)."""
    _patch_torchaudio()  # must run BEFORE importing speechbrain

    # Some Windows corporate environments block move operations used by HF/SpeechBrain.
    # Keep shutil.move's signature intact so callers can pass copy_function.
    import shutil
    original_move = shutil.move

    def _safe_move(src, dst, copy_function=shutil.copy2):
        try:
            return original_move(src, dst, copy_function=copy_function)
        except OSError:
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                return shutil.copytree(src, dst, copy_function=copy_function)
            return copy_function(src, dst)

    shutil.move = _safe_move
    
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    save_dir = str(MODELS_DIR / "speechbrain_ecapa")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=save_dir,
        run_opts={"device": DEVICE},
    )
    return classifier


registry.register("speaker_ecapa", _load_ecapa)


# ─── Task: Speaker Identification ────────────────────────────────────────────

def analyze(audio: AudioData, vad_result=None, **kwargs) -> SpeakerIDResult:
    """
    Extract speaker embedding and return speaker segment.
    Edge edition: ECAPA-TDNN only (no pyannote diarization needed).
    Returns SpeakerIDResult with segment labels and 192-d ECAPA embedding.
    """
    return _ecapa_only(audio)


def _ecapa_only(audio: AudioData) -> SpeakerIDResult:
    embedding = _get_embedding(audio)
    return SpeakerIDResult(
        segments=[SpeakerSegment(
            start_sec=0.0,
            end_sec=audio.duration_sec,
            speaker_id="SPEAKER_00",
        )],
        num_speakers=1,
        embedding=embedding,
    )


def _get_embedding(audio: AudioData) -> Optional[List[float]]:
    try:
        import torch
        classifier = registry.get("speaker_ecapa")
        signal = torch.from_numpy(audio.waveform).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = classifier.encode_batch(signal)
        return emb.squeeze().tolist()
    except Exception as e:
        logger.warning(f"ECAPA embedding failed: {e}")
        return None


# ─── Speaker Verification ─────────────────────────────────────────────────────

def verify_speakers(
    audio1: AudioData,
    audio2: AudioData,
    threshold: float = 0.25,
) -> SpeakerVerificationResult:
    """
    Verify whether two audio clips are from the same speaker using cosine similarity.
    """
    import torch
    import torch.nn.functional as F

    try:
        classifier = registry.get("speaker_ecapa")
        sig1 = torch.from_numpy(audio1.waveform).unsqueeze(0).to(DEVICE)
        sig2 = torch.from_numpy(audio2.waveform).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e1 = classifier.encode_batch(sig1).squeeze()
            e2 = classifier.encode_batch(sig2).squeeze()
        score = float(F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)))
        return SpeakerVerificationResult(
            score=round(score, 4),
            is_same_speaker=score >= threshold,
            threshold=threshold,
        )
    except Exception as e:
        logger.error(f"Speaker verification failed: {e}", exc_info=True)
        return SpeakerVerificationResult(success=False, error=str(e))
