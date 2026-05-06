"""
tasks/music_speech_detection.py  [EDGE EDITION]
================================================
Task 11: Music / Speech / Noise Detection.
Model: ina-foss/inaSpeechSegmenter — lightweight CNN (~5MB)
Labels: speech, music, noise, noEnergy
"""

from __future__ import annotations

import logging
import sys
import tempfile
import os

# Corporate Windows blocks PATH resolution. explicitly point to FFmpeg binary if needed
if "FFMPEG_BINARY" not in os.environ:
    os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe"
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from core.audio_io import AudioData
from core.model_registry import registry, MODELS_DIR
from core.result_schema import MusicSpeechResult, MusicSpeechSegment

logger = logging.getLogger(__name__)

INA_REPO = MODELS_DIR / "inaSpeechSegmenter"


def _load_model():
    """Load inaSpeechSegmenter from local git clone or pip install."""
    # Try local clone first
    if INA_REPO.exists() and str(INA_REPO) not in sys.path:
        sys.path.insert(0, str(INA_REPO))

    try:
        from inaSpeechSegmenter import Segmenter
        seg = Segmenter(vad_engine="smn", detect_gender=False)
        logger.info("inaSpeechSegmenter loaded successfully.")
        return seg
    except ImportError:
        logger.warning("inaSpeechSegmenter not found. Install via: pip install inaspeechsegmenter")
        raise


registry.register("music_speech", _load_model)


def analyze(audio: AudioData, **kwargs) -> MusicSpeechResult:
    """
    Segment audio into speech / music / noise / noEnergy regions.
    Returns MusicSpeechResult with per-segment labels and summary fractions.
    """
    try:
        seg = registry.get("music_speech")

        # inaSpeechSegmenter needs a WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio.waveform, audio.sample_rate)
            tmp_path = tmp.name

        try:
            seg_output = seg(tmp_path)
        finally:
            os.unlink(tmp_path)

        segments: List[MusicSpeechSegment] = []
        speech_sec = music_sec = noise_sec = 0.0

        for label, start, end in seg_output:
            segments.append(MusicSpeechSegment(
                start_sec=round(float(start), 3),
                end_sec=round(float(end), 3),
                label=label,
            ))
            dur = end - start
            if label == "speech":
                speech_sec += dur
            elif label == "music":
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
