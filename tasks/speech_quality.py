"""
tasks/speech_quality.py
=======================
Task 5: Speech Quality & Intelligibility Assessment.
Models:
  - DNSMOS P.835 (Microsoft DNS-Challenge) — non-intrusive MOS prediction
  - PESQ (wideband) — intrusive quality (needs clean reference)
  - STOI — Short-Time Objective Intelligibility
  - SNR estimation via energy analysis
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from core.audio_io import AudioData
from core.feature_extractor import get_snr
from core.model_registry import registry, DEVICE, MODELS_DIR
from core.result_schema import SpeechQualityResult

logger = logging.getLogger(__name__)

DNSMOS_REPO = MODELS_DIR / "DNS-Challenge"
DNSMOS_MODEL_PATH = DNSMOS_REPO / "DNSMOS" / "DNSMOS" / "model_v8.onnx"
DNSMOS_PRI_MODEL_PATH = DNSMOS_REPO / "DNSMOS" / "DNSMOS" / "sig_bak_ovr.onnx"

SR_DNSMOS = 16_000


def _load_dnsmos():
    """Load DNSMOS ONNX models from local DNS-Challenge clone."""
    try:
        import onnxruntime as ort

        if not DNSMOS_MODEL_PATH.exists():
            logger.warning(
                f"DNSMOS model not found at {DNSMOS_MODEL_PATH}. "
                "Run download_models.py first. Using SNR-only quality estimation."
            )
            return None

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        sess_ovr = ort.InferenceSession(str(DNSMOS_MODEL_PATH), sess_options=opts)
        sess_pri = ort.InferenceSession(str(DNSMOS_PRI_MODEL_PATH), sess_options=opts)
        return {"sess_ovr": sess_ovr, "sess_pri": sess_pri}
    except ImportError:
        logger.warning("onnxruntime not installed. Install with: pip install onnxruntime")
        return None
    except Exception as e:
        logger.warning(f"DNSMOS load error: {e}")
        return None


registry.register("dnsmos", _load_dnsmos)


def _compute_mel_features(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Compute log-mel features for DNSMOS sig_bak_ovr.onnx (rank-3 input)."""
    import librosa
    if sr != SR_DNSMOS:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SR_DNSMOS)
    clip_len = int(9.01 * SR_DNSMOS)
    if len(waveform) < clip_len:
        waveform = np.pad(waveform, (0, clip_len - len(waveform)))
    else:
        waveform = waveform[:clip_len]
    waveform = waveform.astype(np.float32)
    hop = int(0.010 * SR_DNSMOS)
    win = int(0.020 * SR_DNSMOS)
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SR_DNSMOS, n_fft=512,
        hop_length=hop, win_length=win, n_mels=120, fmin=20, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max) + 40.0
    # (120, T) -> (1, T, 120) — trim/pad to exactly 900 frames (DNSMOS requirement)
    frames = log_mel.T  # (T, 120)
    if frames.shape[0] > 900:
        frames = frames[:900, :]
    elif frames.shape[0] < 900:
        pad = np.zeros((900 - frames.shape[0], 120), dtype=np.float32)
        frames = np.vstack([frames, pad])
    return frames[np.newaxis, ...].astype(np.float32)  # (1, 900, 120)


def _run_dnsmos(waveform: np.ndarray, sr: int, sessions: dict) -> dict:
    """Run DNSMOS — model_v8.onnx needs raw audio, sig_bak_ovr needs mel features."""
    try:
        import librosa
        # Prepare 16kHz audio, exactly 9.01s = 144160 samples
        if sr != SR_DNSMOS:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SR_DNSMOS)
        target_len = int(9.01 * SR_DNSMOS)  # 144160
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        else:
            waveform = waveform[:target_len]
        raw_audio = waveform.astype(np.float32)

        # Mel features for sig_bak_ovr.onnx
        mel_feats = _compute_mel_features(raw_audio, SR_DNSMOS)  # (1, 900, 120)

        def _infer(session, raw, mel):
            inp = session.get_inputs()[0]
            name = inp.name
            rank = len(inp.shape)
            expected_size = inp.shape[1] if len(inp.shape) > 1 else None
            if rank == 3:
                data = mel           # mel features
            elif rank == 2 and expected_size and expected_size > 10000:
                data = raw.reshape(1, -1)   # raw audio (1, 144160)
            else:
                data = mel.reshape(1, -1)   # flat mel fallback
            return session.run(None, {name: data})[0].flatten()

        ovr = _infer(sessions["sess_ovr"], raw_audio, mel_feats)
        pri = _infer(sessions["sess_pri"], raw_audio, mel_feats)
        result = {"dnsmos_overall": float(np.clip(ovr[0], 1.0, 5.0))}
        if len(pri) >= 3:
            result["dnsmos_noisy"]      = float(np.clip(pri[0], 1.0, 5.0))
            result["dnsmos_signal"]     = float(np.clip(pri[1], 1.0, 5.0))
            result["dnsmos_background"] = float(np.clip(pri[2], 1.0, 5.0))
        return result
    except Exception as e:
        logger.warning(f"DNSMOS inference error: {e}")
        return {}




def analyze(
    audio: AudioData,
    clean_reference: Optional[np.ndarray] = None,
    **kwargs,
) -> SpeechQualityResult:
    """
    Assess speech quality and intelligibility.
    clean_reference: optional clean speech array for PESQ/STOI (intrusive metrics).
    """
    snr = get_snr(audio.waveform, audio.sample_rate)
    dnsmos_scores = {}

    # DNSMOS (non-intrusive)
    try:
        sessions = registry.get("dnsmos")
        if sessions:
            dnsmos_scores = _run_dnsmos(audio.waveform, audio.sample_rate, sessions)
    except Exception as e:
        logger.warning(f"DNSMOS skipped: {e}")

    # PESQ (intrusive — only if clean reference provided)
    pesq_score = None
    if clean_reference is not None:
        try:
            from pesq import pesq
            ref = clean_reference
            deg = audio.waveform
            # PESQ needs same length
            min_len = min(len(ref), len(deg))
            pesq_score = float(pesq(audio.sample_rate, ref[:min_len], deg[:min_len], "wb"))
        except ImportError:
            logger.warning("pesq not installed: pip install pesq")
        except Exception as e:
            logger.warning(f"PESQ failed: {e}")

    # STOI (intrusive)
    stoi_score = None
    if clean_reference is not None:
        try:
            from pystoi import stoi
            ref = clean_reference
            deg = audio.waveform
            min_len = min(len(ref), len(deg))
            stoi_score = float(stoi(ref[:min_len], deg[:min_len], audio.sample_rate, extended=False))
        except ImportError:
            logger.warning("pystoi not installed: pip install pystoi")
        except Exception as e:
            logger.warning(f"STOI failed: {e}")

    return SpeechQualityResult(
        dnsmos_overall=round(dnsmos_scores.get("dnsmos_overall", 0.0), 3),
        dnsmos_signal=round(dnsmos_scores.get("dnsmos_signal", 0.0), 3),
        dnsmos_background=round(dnsmos_scores.get("dnsmos_background", 0.0), 3),
        dnsmos_noisy=round(dnsmos_scores.get("dnsmos_noisy", 0.0), 3),
        pesq_score=round(pesq_score, 3) if pesq_score is not None else None,
        stoi_score=round(stoi_score, 3) if stoi_score is not None else None,
        snr_db=round(snr, 2),
    )
