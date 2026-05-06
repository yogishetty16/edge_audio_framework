"""
tasks/asr.py  [EDGE EDITION]
============================
Task 1: Automatic Speech Recognition.
Model: faster-whisper tiny.en (CTranslate2 INT8, ~39MB)
~4x faster than openai-whisper, runs real-time on CPU.
Also supports faster-whisper base/small for better accuracy.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.audio_io import AudioData
from core.model_registry import registry, DEVICE, MODELS_DIR
from core.result_schema import ASRResult

logger = logging.getLogger(__name__)

# Model size: tiny.en (39MB) | base.en (74MB) | small.en (244MB) | base (74MB Multilingual)
# Adjust WHISPER_SIZE based on your accuracy/speed trade-off
WHISPER_SIZE = "base"
WHISPER_COMPUTE = "int8"          # int8 | float16 | float32
WHISPER_BEAM_SIZE = 1             # 1 = greedy (fastest for edge)


def _load_model():
    from faster_whisper import WhisperModel

    # CTranslate2 download location
    model_dir = str(MODELS_DIR / f"faster-whisper-{WHISPER_SIZE}")
    device = "cpu"
    if DEVICE == "cuda":
        device = "cuda"

    logger.info(f"Loading faster-whisper {WHISPER_SIZE} (compute={WHISPER_COMPUTE}) …")
    model = WhisperModel(
        WHISPER_SIZE,
        device=device,
        compute_type=WHISPER_COMPUTE,
        download_root=str(MODELS_DIR),
        num_workers=1,
        cpu_threads=4,
    )
    return model


registry.register("asr", _load_model)


def analyze(audio: AudioData, vad_result=None, **kwargs) -> ASRResult:
    """
    Transcribe speech. Uses VAD segments if available (skips silence → faster).
    Returns ASRResult with text, detected language, and segments.
    """
    try:
        model = registry.get("asr")
        vad_threshold = kwargs.get("threshold", 0.5)
        waveform = audio.waveform.astype(np.float32)

        # Build VAD-based initial_prompt_tokens (only transcribe speech regions)
        # faster-whisper accepts a numpy float32 waveform directly
        segments_iter, info = model.transcribe(
            waveform,
            language=None,          # auto-detect
            beam_size=WHISPER_BEAM_SIZE,
            best_of=1,
            temperature=0.0,        # greedy decode (fastest)
            condition_on_previous_text=False, # Prevents getting stuck in hallucination loops
            logprob_threshold=-1.0,           # Discard garbage/noise transcriptions
            compression_ratio_threshold=2.4,  # Automatically discard repeating text like "88888888"
            no_speech_threshold=0.6,          # Stricter check for pure noise
            word_timestamps=False,
            vad_filter=True,        # built-in VAD filter skips silence
            vad_parameters={
                "threshold": vad_threshold,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 100,
            },
        )

        segments = list(segments_iter)
        full_text = " ".join(s.text.strip() for s in segments)

        seg_dicts = [
            {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
            for s in segments
        ]

        return ASRResult(
            text=full_text.strip(),
            language=info.language,
            segments=seg_dicts,
        )

    except ImportError:
        logger.error("faster-whisper not installed. Run: pip install faster-whisper")
        return ASRResult(success=False, error="faster-whisper not installed")
    except Exception as e:
        logger.error(f"ASR failed: {e}", exc_info=True)
        return ASRResult(success=False, error=str(e))
