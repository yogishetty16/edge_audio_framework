"""Lightweight impulse event detector.

This task catches short, sharp acoustic events such as gunshot-like pops,
impact sounds, or explosions using signal features. It is intentionally model
free so it can run quickly on edge devices and help when large ESC models miss
brief transients.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from core.audio_io import AudioData

logger = logging.getLogger(__name__)


@dataclass
class ImpulseEventResult:
    success: bool = True
    error: str | None = None
    is_impulse: bool = False
    event_label: str = "normal_audio"
    confidence: float = 0.0
    peak_amplitude: float = 0.0
    rms: float = 0.0
    crest_factor: float = 0.0
    transient_ratio: float = 0.0
    high_band_ratio: float = 0.0
    zero_crossing_rate: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__.copy()


def analyze(audio: AudioData, **kwargs) -> ImpulseEventResult:
    """Detect short high-energy impulse events in an audio window."""
    try:
        waveform = np.asarray(audio.waveform, dtype=np.float32)
        if waveform.size == 0:
            return ImpulseEventResult(success=False, error="empty audio")

        waveform = np.nan_to_num(waveform)
        abs_wav = np.abs(waveform)
        peak = float(abs_wav.max())
        rms = float(np.sqrt(np.mean(np.square(waveform))) + 1e-9)
        crest = float(peak / rms)

        frame_size = max(int(audio.sample_rate * 0.02), 128)
        hop = max(frame_size // 2, 1)
        frame_rms = _frame_rms(waveform, frame_size, hop)
        if frame_rms.size == 0:
            return ImpulseEventResult()

        median_energy = float(np.median(frame_rms) + 1e-9)
        peak_frame_energy = float(frame_rms.max())
        transient_ratio = float(peak_frame_energy / median_energy)
        zcr = _zero_crossing_rate(waveform)
        high_band_ratio = _high_band_ratio(waveform, audio.sample_rate)

        confidence = _score_impulse(
            crest_factor=crest,
            transient_ratio=transient_ratio,
            peak_amplitude=peak,
            high_band_ratio=high_band_ratio,
        )
        # Gunshots on laptop mics/phones are often distorted or compressed. 
        # Lower threshold from 0.62 to 0.40 to ensure we catch them.
        is_impulse = confidence >= 0.40
        label = "gunshot_or_impact_like_impulse" if is_impulse else "normal_audio"

        return ImpulseEventResult(
            is_impulse=is_impulse,
            event_label=label,
            confidence=round(confidence, 4),
            peak_amplitude=round(peak, 4),
            rms=round(rms, 6),
            crest_factor=round(crest, 3),
            transient_ratio=round(transient_ratio, 3),
            high_band_ratio=round(high_band_ratio, 4),
            zero_crossing_rate=round(zcr, 4),
            features={
                "peak_frame_energy": round(peak_frame_energy, 6),
                "median_frame_energy": round(median_energy, 6),
            },
        )
    except Exception as exc:
        logger.error(f"Impulse event detection failed: {exc}", exc_info=True)
        return ImpulseEventResult(success=False, error=str(exc))


def _frame_rms(waveform: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    if len(waveform) < frame_size:
        return np.array([np.sqrt(np.mean(np.square(waveform)))], dtype=np.float32)

    values = []
    for start in range(0, len(waveform) - frame_size + 1, hop):
        frame = waveform[start:start + frame_size]
        values.append(np.sqrt(np.mean(np.square(frame))))
    return np.asarray(values, dtype=np.float32)


def _zero_crossing_rate(waveform: np.ndarray) -> float:
    if waveform.size < 2:
        return 0.0
    signs = np.signbit(waveform)
    return float(np.mean(signs[1:] != signs[:-1]))


def _high_band_ratio(waveform: np.ndarray, sample_rate: int) -> float:
    if waveform.size < 256:
        return 0.0

    window = np.hanning(waveform.size)
    spectrum = np.abs(np.fft.rfft(waveform * window)) ** 2
    freqs = np.fft.rfftfreq(waveform.size, d=1.0 / sample_rate)
    total = float(spectrum.sum() + 1e-12)
    high = float(spectrum[freqs >= 2500].sum())
    return high / total


def _score_impulse(
    crest_factor: float,
    transient_ratio: float,
    peak_amplitude: float,
    high_band_ratio: float,
) -> float:
    score = 0.0
    # A crest factor > 8 on a laptop mic is already a massive transient
    score += min(max((crest_factor - 5.0) / 10.0, 0.0), 1.0) * 0.40
    # A transient ratio > 6 is a very sharp spike compared to background
    score += min(max((transient_ratio - 5.0) / 10.0, 0.0), 1.0) * 0.40
    # Peak amplitude: anything moderately loud adds score
    score += min(max((peak_amplitude - 0.05) / 0.40, 0.0), 1.0) * 0.10
    # High frequency ratio: laptop mics compress this heavily, require less
    score += min(max((high_band_ratio - 0.05) / 0.20, 0.0), 1.0) * 0.10
    return min(score, 1.0)
