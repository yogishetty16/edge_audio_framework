"""
tasks/anomaly_detection.py  [EDGE EDITION]
==========================================
Task 12: Anomaly Detection in Audio Streams.
Method: PANNs ONNX embeddings + sklearn IsolationForest.
No large model required — uses embeddings from the AED model.
The IsolationForest learns what "normal" sounds like and flags deviations.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.audio_io import AudioData, split_into_chunks
from core.model_registry import registry, DEVICE, MODELS_DIR
from core.result_schema import AnomalyResult

logger = logging.getLogger(__name__)

# Trained IsolationForest saved here after baseline fitting
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_isolation_forest.pkl"
PANNS_ONNX_PATH = MODELS_DIR / "panns_mobilenetv2.onnx"
PANNS_SR = 32_000

# Anomaly threshold: scores below this => anomalous
# IsolationForest score < 0 = anomaly (sklearn convention)
ANOMALY_THRESHOLD = -0.1


def _load_embedder():
    """Load PANNs ONNX just for embeddings (reuse AED session if already loaded)."""
    try:
        from core.onnx_utils import create_onnx_session
        if PANNS_ONNX_PATH.exists():
            session = create_onnx_session(str(PANNS_ONNX_PATH))
            return {"type": "onnx", "session": session}
    except Exception:
        pass

    # Fallback: use raw spectral features
    logger.info("PANNs ONNX not found - using conservative signal heuristics for anomaly fallback")
    return {"type": "spectrogram"}


registry.register("anomaly_embedder", _load_embedder)


def _get_embedding(audio: AudioData, bundle: dict) -> np.ndarray:
    """Extract a fixed-size embedding vector from audio."""
    if bundle["type"] == "onnx":
        wv = _pad_resample(audio)
        inp_name = bundle["session"].get_inputs()[0].name
        out = bundle["session"].run(None, {inp_name: wv})[0]
        return out.flatten()
    else:
        # Fallback: mean + std of mel-spectrogram as features
        from core.feature_extractor import get_melspectrogram
        mel = get_melspectrogram(audio.waveform, audio.sample_rate, as_db=True)
        return np.concatenate([mel.mean(axis=1), mel.std(axis=1)])


def _pad_resample(audio: AudioData) -> np.ndarray:
    import librosa
    wv = audio.waveform
    if audio.sample_rate != PANNS_SR:
        wv = librosa.resample(wv, orig_sr=audio.sample_rate, target_sr=PANNS_SR)
    return wv.reshape(1, -1).astype(np.float32)


def fit_baseline(audio_files: List[str], save: bool = True):
    """
    Train IsolationForest on 'normal' audio files.
    Call this once before using anomaly detection.

    Parameters
    ----------
    audio_files : List of paths to normal/baseline audio files.
    save        : Save fitted model to ANOMALY_MODEL_PATH.
    """
    from sklearn.ensemble import IsolationForest
    from core.audio_io import load_audio

    bundle = registry.get("anomaly_embedder")
    embeddings = []

    for f in audio_files:
        try:
            audio = load_audio(f)
            for chunk in split_into_chunks(audio, chunk_duration_sec=2.0):
                emb = _get_embedding(chunk, bundle)
                embeddings.append(emb)
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    if not embeddings:
        raise ValueError("No embeddings extracted from baseline audio files.")

    X = np.stack(embeddings)
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # assume 5% anomaly rate
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    logger.info(f"IsolationForest trained on {len(embeddings)} clips.")

    if save:
        ANOMALY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ANOMALY_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Anomaly model saved: {ANOMALY_MODEL_PATH}")

    return model


def _load_iforest() -> Optional[Any]:
    if ANOMALY_MODEL_PATH.exists():
        with open(ANOMALY_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def analyze(audio: AudioData, **kwargs) -> AnomalyResult:
    """
    Score audio for anomalies using IsolationForest on PANNs embeddings.
    Returns AnomalyResult with anomaly score and flag.

    Note: If no baseline model is fitted yet, scores using distance from
    mean embedding as a fallback heuristic.
    """
    try:
        bundle = registry.get("anomaly_embedder")
        iforest = _load_iforest()

        # Process in 2s chunks
        chunks = split_into_chunks(audio, chunk_duration_sec=2.0, overlap_sec=0.25)
        if not chunks:
            chunks = [audio]

        chunk_scores: List[float] = []
        anomalous_segs: List[Dict] = []

        for i, chunk in enumerate(chunks):
            emb = _get_embedding(chunk, bundle).reshape(1, -1)

            if iforest is not None:
                # sklearn IsolationForest: score_samples returns negative values for anomalies
                score = float(iforest.score_samples(emb)[0])
                is_anom = score < ANOMALY_THRESHOLD
            else:
                # Conservative untrained fallback: do not treat embedding magnitude
                # as an anomaly. Without a fitted baseline, only obvious signal
                # defects should raise an anomaly flag.
                score, is_anom = _score_untrained_fallback(chunk)

            chunk_scores.append(score)
            if is_anom:
                start = chunk.source_path and 0 or i * 2.0
                anomalous_segs.append({
                    "chunk_idx": i,
                    "start_sec": round(i * 2.0, 2),
                    "end_sec": round(min((i + 1) * 2.0, audio.duration_sec), 2),
                    "score": round(score, 4),
                })

        overall_score = float(np.mean(chunk_scores)) if chunk_scores else 0.0
        is_anomaly = overall_score < ANOMALY_THRESHOLD

        return AnomalyResult(
            anomaly_score=round(overall_score, 4),
            is_anomaly=is_anomaly,
            threshold=ANOMALY_THRESHOLD,
            anomalous_segments=anomalous_segs,
        )

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}", exc_info=True)
        return AnomalyResult(success=False, error=str(e))


def _score_untrained_fallback(audio: AudioData) -> tuple[float, bool]:
    """Conservative anomaly score when no baseline model is available."""
    try:
        from core.feature_extractor import detect_clipping, get_rms_db, get_silence_fraction

        clipping = detect_clipping(audio.waveform)
        rms_db = get_rms_db(audio.waveform)
        silence_fraction = get_silence_fraction(audio.waveform)

        if clipping:
            return -0.25, True
        if rms_db > -3.0:
            return -0.18, True
        if silence_fraction > 0.98:
            return -0.12, True
        if rms_db < -60.0:
            return -0.11, True
        return 0.0, False
    except Exception:
        return 0.0, False
