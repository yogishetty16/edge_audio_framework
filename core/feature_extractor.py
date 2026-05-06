"""
core/feature_extractor.py
=========================
Signal-level audio feature utilities.
No ML models — pure numpy/scipy/librosa.
"""
from __future__ import annotations

import numpy as np


def get_snr(waveform: np.ndarray, sample_rate: int) -> float:
    """
    Estimate Signal-to-Noise Ratio (dB) using a simple energy-based approach.
    Assumes the bottom 20% quietest frames are 'noise', rest is 'signal'.
    """
    frame_size = 1024
    hop = 512
    waveform = waveform.astype(np.float32)

    if len(waveform) < frame_size:
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        return 0.0 if rms < 1e-10 else 20.0

    # Frame-level RMS energy
    n_frames = (len(waveform) - frame_size) // hop + 1
    frame_energies = np.array([
        np.mean(waveform[i * hop: i * hop + frame_size] ** 2)
        for i in range(n_frames)
    ])

    if len(frame_energies) == 0 or frame_energies.max() < 1e-12:
        return 0.0

    threshold = np.percentile(frame_energies, 20)
    noise_energy = frame_energies[frame_energies <= threshold].mean()
    signal_energy = frame_energies[frame_energies > threshold].mean() if np.any(frame_energies > threshold) else threshold

    if noise_energy < 1e-12:
        return 60.0  # effectively noiseless
    snr = 10.0 * np.log10(signal_energy / noise_energy)
    return float(np.clip(snr, -10.0, 60.0))


def get_rms_db(waveform: np.ndarray) -> float:
    """Return RMS level in dBFS."""
    rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
    if rms < 1e-10:
        return -100.0
    return float(20.0 * np.log10(rms))


def get_dc_offset(waveform: np.ndarray) -> float:
    """Return mean DC offset of the signal."""
    return float(np.mean(waveform))


def detect_clipping(waveform: np.ndarray, threshold: float = 0.99) -> bool:
    """Return True if any sample exceeds the clipping threshold."""
    return bool(np.any(np.abs(waveform) >= threshold))


def get_silence_fraction(
    waveform: np.ndarray,
    frame_size: int = 1024,
    threshold: float = 0.01,
) -> float:
    """
    Return fraction of frames below the RMS energy threshold (silence).
    """
    waveform = waveform.astype(np.float32)
    if len(waveform) < frame_size:
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        return 1.0 if rms < threshold else 0.0

    n_frames = len(waveform) // frame_size
    frames = waveform[:n_frames * frame_size].reshape(n_frames, frame_size)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))
    silent = np.sum(rms_per_frame < threshold)
    return float(silent / n_frames)


def get_melspectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 512,
    as_db: bool = False,
) -> np.ndarray:
    """
    Compute mel-spectrogram. Returns array of shape (n_mels, T).
    """
    import librosa
    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float32),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    if as_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel
