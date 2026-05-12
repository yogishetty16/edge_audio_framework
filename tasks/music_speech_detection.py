"""
tasks/music_speech_detection.py
================================
Task 11: Music / Speech / Silence Detection.
Model: Custom DSP (zero-crossing rate + spectral features)
Labels: speech, music, silence
"""

from __future__ import annotations

import logging
from typing import List
import numpy as np

from core.audio_io import AudioData
from core.model_registry import registry
from core.result_schema import MusicSpeechResult, MusicSpeechSegment

logger = logging.getLogger(__name__)


def _load_model():
    """No external model required. Using built-in librosa DSP."""
    import librosa
    return librosa


registry.register("music_speech", _load_model)


def analyze(audio: AudioData, **kwargs) -> MusicSpeechResult:
    """
    Segment audio into speech / music / silence regions using simple DSP heuristics.
    Returns MusicSpeechResult with per-segment labels and summary fractions.
    """
    try:
        librosa = registry.get("music_speech")
        waveform = audio.waveform
        sr = audio.sample_rate

        # Simple energy-based silence detection
        rms = librosa.feature.rms(y=waveform)[0]
        threshold = 0.01  # heuristic threshold for silence
        is_silence = rms < threshold

        # Zero-crossing rate (Speech typically has higher variance in ZCR than music)
        zcr = librosa.feature.zero_crossing_rate(y=waveform)[0]
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]

        # Simple heuristic classification per frame
        frames = len(rms)
        frame_duration = len(waveform) / sr / frames if frames > 0 else 0.023

        segments: List[MusicSpeechSegment] = []
        speech_sec = music_sec = noise_sec = 0.0
        
        current_label = None
        start_time = 0.0

        for i in range(frames):
            if is_silence[i]:
                label = "silence"
            else:
                # High ZCR variance often indicates unvoiced speech consonants
                # High centroid often indicates music cymbals/high notes
                # This is a basic heuristic for edge devices without deep learning
                if zcr[i] > 0.1 or centroid[i] > 3000:
                    label = "speech"
                else:
                    label = "music"

            if current_label is None:
                current_label = label
                start_time = i * frame_duration
            elif current_label != label:
                end_time = i * frame_duration
                segments.append(MusicSpeechSegment(
                    start_sec=round(float(start_time), 3),
                    end_sec=round(float(end_time), 3),
                    label=current_label,
                ))
                dur = end_time - start_time
                if current_label == "speech":
                    speech_sec += dur
                elif current_label == "music":
                    music_sec += dur
                else:
                    noise_sec += dur
                
                current_label = label
                start_time = i * frame_duration

        # Last segment
        if current_label is not None:
            end_time = frames * frame_duration
            segments.append(MusicSpeechSegment(
                start_sec=round(float(start_time), 3),
                end_sec=round(float(end_time), 3),
                label=current_label,
            ))
            dur = end_time - start_time
            if current_label == "speech":
                speech_sec += dur
            elif current_label == "music":
                music_sec += dur
            else:
                noise_sec += dur

        total = max(audio.duration_sec, 1e-6)
        return MusicSpeechResult(
            segments=segments,
            speech_fraction=round(speech_sec / total, 4),
            music_fraction=round(music_sec / total, 4),
            noise_fraction=round(noise_sec / total, 4),
        )

    except Exception as e:
        logger.error(f"Music/Speech detection failed: {e}", exc_info=True)
        return MusicSpeechResult(success=False, error=str(e))
