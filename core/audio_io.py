"""
core/audio_io.py
================
Audio data container and I/O utilities.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class AudioData:
    """Central audio container passed between all pipeline tasks."""
    waveform: np.ndarray        # float32, shape (N,)
    sample_rate: int            # Hz, e.g. 16000
    duration_sec: float         # len(waveform) / sample_rate
    n_channels: int = 1
    source_path: Optional[str] = None

    def __post_init__(self):
        self.waveform = self.waveform.astype(np.float32)
        if self.duration_sec == 0 and self.sample_rate > 0:
            self.duration_sec = len(self.waveform) / self.sample_rate


def load_audio(path: str, target_sr: int = 16000) -> AudioData:
    """Load any audio file and resample to target_sr."""
    import soundfile as sf
    import librosa

    path = str(path)
    try:
        data, sr = sf.read(path, always_2d=False)
    except Exception:
        # Fallback to librosa for mp3/m4a/etc.
        data, sr = librosa.load(path, sr=None, mono=True)

    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return AudioData(
        waveform=data,
        sample_rate=sr,
        duration_sec=len(data) / sr,
        n_channels=1,
        source_path=path,
    )


def split_into_chunks(
    audio: AudioData,
    chunk_duration_sec: float,
    overlap_sec: float = 0.0,
) -> List[AudioData]:
    """Split AudioData into fixed-size chunks with optional overlap."""
    sr = audio.sample_rate
    chunk_samples = int(chunk_duration_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step_samples = max(chunk_samples - overlap_samples, 1)

    chunks: List[AudioData] = []
    waveform = audio.waveform
    total = len(waveform)
    start = 0

    while start < total:
        end = min(start + chunk_samples, total)
        chunk_wave = waveform[start:end]
        # Skip very short trailing chunks (< 0.1s)
        if len(chunk_wave) < int(0.1 * sr):
            break
        chunks.append(AudioData(
            waveform=chunk_wave.copy(),
            sample_rate=sr,
            duration_sec=len(chunk_wave) / sr,
            n_channels=audio.n_channels,
            source_path=audio.source_path,
        ))
        start += step_samples

    return chunks
