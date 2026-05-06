"""
tasks/audio_quality_monitor.py
===============================
Task 10: Audio Quality Monitoring.
Composite analysis using:
  - DNSMOS MOS score (from speech_quality task)
  - SNR, RMS level, DC offset, clipping detection, silence fraction
  - Produces a quality label: excellent / good / fair / poor
"""

from __future__ import annotations

import logging

import numpy as np

from core.audio_io import AudioData
from core.feature_extractor import (
    get_snr, get_rms_db, get_dc_offset,
    detect_clipping, get_silence_fraction,
)
from core.result_schema import AudioQualityResult

logger = logging.getLogger(__name__)


def _mos_to_label(mos: float, snr: float) -> str:
    """Map composite score to quality label."""
    if mos >= 4.0 and snr >= 20:
        return "excellent"
    elif mos >= 3.5 or snr >= 15:
        return "good"
    elif mos >= 2.5 or snr >= 8:
        return "fair"
    else:
        return "poor"


def analyze(audio: AudioData, **kwargs) -> AudioQualityResult:
    """
    Comprehensive audio quality monitoring.
    Returns AudioQualityResult with MOS, SNR, clipping status, silence fraction, and label.
    """
    try:
        waveform = audio.waveform
        sr = audio.sample_rate

        snr = get_snr(waveform, sr)
        rms_db = get_rms_db(waveform)
        dc_offset = get_dc_offset(waveform)
        clipping = detect_clipping(waveform)
        silence_frac = get_silence_fraction(waveform)

        # Try to get MOS from DNSMOS
        mos = 0.0
        try:
            from core.model_registry import registry
            sessions = registry.get("dnsmos")
            if sessions:
                from tasks.speech_quality import _run_dnsmos
                dnsmos = _run_dnsmos(waveform, sr, sessions)
                mos = dnsmos.get("dnsmos_overall", 0.0)
        except Exception:
            # Estimate MOS from SNR as fallback
            mos = min(5.0, max(1.0, 1.0 + snr / 10.0))

        quality_label = _mos_to_label(mos, snr)

        return AudioQualityResult(
            mos=round(mos, 3),
            snr_db=round(snr, 2),
            clipping_detected=clipping,
            dc_offset=round(dc_offset, 6),
            rms_db=round(rms_db, 2),
            silence_fraction=round(silence_frac, 4),
            quality_label=quality_label,
        )

    except Exception as e:
        logger.error(f"Audio quality monitoring failed: {e}", exc_info=True)
        return AudioQualityResult(success=False, error=str(e))
