"""
tasks/vad.py  [EDGE EDITION]
============================
Task 6: Voice Activity Detection -- Silero VAD ONNX.
Model: Silero VAD (2MB ONNX model, <1ms per 30ms chunk).
Supports both streaming (chunk) and batch (full-file) modes.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from core.audio_io import AudioData
from core.model_registry import registry, DEVICE, MODELS_DIR
from core.result_schema import VADResult, VADSegment

logger = logging.getLogger(__name__)

SILERO_SR = 16_000
THRESHOLD = 0.5
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 100


def _find_silero_onnx():
    """Find the silero_vad.onnx file in local paths."""
    import glob
    # Path to local silero-vad ONNX file from git clone
    SILERO_LOCAL = MODELS_DIR / "silero-vad" / "files" / "silero_vad.onnx"
    # Fallback: torch hub clones it here
    SILERO_HUB_SEARCH = [
        MODELS_DIR / "torch_hub" / "hub" / "snakers4_silero-vad_master" / "files" / "silero_vad.onnx",
        MODELS_DIR / "torch_hub" / "hub" / "snakers4_silero-vad_v5.1" / "files" / "silero_vad.onnx",
    ]
    
    if SILERO_LOCAL.exists():
        return str(SILERO_LOCAL)
    for p in SILERO_HUB_SEARCH:
        if p.exists():
            return str(p)
    pattern = str(MODELS_DIR / "torch_hub" / "**" / "silero_vad.onnx")
    found = glob.glob(pattern, recursive=True)
    if found:
        return found[0]
    return None


def _load_model():
    """Load Silero VAD -- tries local ONNX first, raises error if not found."""
    import torch
    import sys

    onnx_path = _find_silero_onnx()

    if onnx_path:
        logger.info(f"Loading Silero VAD from local ONNX: {onnx_path}")
        from pathlib import Path
        # Look for hubconf.py in parent directories of onnx_path
        p = Path(onnx_path).parent
        hubconf_dir = None
        while p != p.parent:
            if (p / "hubconf.py").exists():
                hubconf_dir = p
                break
            p = p.parent
        
        # Fallback to MODELS_DIR / "silero-vad"
        if not hubconf_dir:
            silero_dir = MODELS_DIR / "silero-vad"
            if (silero_dir / "hubconf.py").exists():
                hubconf_dir = silero_dir

        if hubconf_dir:
            if str(hubconf_dir) not in sys.path:
                sys.path.insert(0, str(hubconf_dir))
            try:
                model, utils = torch.hub.load(
                    repo_or_dir=str(hubconf_dir),
                    model="silero_vad",
                    force_reload=False,
                    onnx=True,
                    source="local",
                    trust_repo=True,
                )
                return {"model": model, "utils": utils}
            except Exception as e:
                logger.info(f"Local hub interface unavailable, using direct ONNX VAD: {e}")
        return _load_onnx_direct(onnx_path)
    else:
        logger.error(
            "Silero VAD ONNX not found locally. Expected at: "
            f"{MODELS_DIR / 'silero-vad' / 'files' / 'silero_vad.onnx'} or "
            f"{MODELS_DIR / 'torch_hub'} -- download with: python download_models.py --task vad"
        )
        raise FileNotFoundError(
            f"Silero VAD ONNX model not found in {MODELS_DIR}. "
            "Run: python download_models.py --task vad"
        )


def _load_onnx_direct(onnx_path: str):
    """Direct ONNX session load fallback for VAD."""
    import onnxruntime as ort
    import torch

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(onnx_path, sess_options=opts,
                                   providers=["CPUExecutionProvider"])

    class OnnxVADModel:
        def __init__(self, sess):
            self.sess = sess
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

        def __call__(self, x, sr, h=None, c=None):
            h = h.numpy() if h is not None else self._h
            c = c.numpy() if c is not None else self._c
            if hasattr(x, 'numpy'): x = x.numpy()
            ort_inputs = {"input": x, "h": h, "c": c,
                          "sr": np.array(sr, dtype=np.int64)}
            out, hn, cn = self.sess.run(["output", "hn", "cn"], ort_inputs)
            return torch.tensor(out), torch.tensor(hn), torch.tensor(cn)

        def reset_states(self):
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    model = OnnxVADModel(session)

    def get_speech_timestamps(waveform, model, threshold=0.5, sampling_rate=16000,
                              min_speech_duration_ms=250, min_silence_duration_ms=100,
                              return_seconds=False):
        """Simplified speech timestamp extraction."""
        window = 512
        wav = waveform.numpy() if hasattr(waveform, 'numpy') else waveform
        timestamps = []
        in_speech = False; speech_start = 0; silence_count = 0
        min_speech = int(min_speech_duration_ms * sampling_rate / 1000 / window)
        min_silence = int(min_silence_duration_ms * sampling_rate / 1000 / window)
        h = np.zeros((2,1,64), np.float32); c = np.zeros((2,1,64), np.float32)
        for i in range(0, len(wav) - window, window):
            chunk = wav[i:i+window].reshape(1, -1).astype(np.float32)
            ort_in = {"input": chunk, "h": h, "c": c,
                      "sr": np.array(sampling_rate, dtype=np.int64)}
            try:
                prob, h, c = session.run(["output","hn","cn"], ort_in)
                p = float(prob.flatten()[0])
            except Exception:
                p = 0.0
            if p >= threshold and not in_speech:
                in_speech = True; speech_start = i; silence_count = 0
            elif p < threshold and in_speech:
                silence_count += 1
                if silence_count >= min_silence:
                    in_speech = False
                    dur = (i - speech_start)
                    if dur >= min_speech * window:
                        s = speech_start / sampling_rate
                        e = i / sampling_rate
                        timestamps.append({"start": s if return_seconds else speech_start,
                                           "end": e if return_seconds else i})
        if in_speech:
            e_s = len(wav) / sampling_rate
            s_s = speech_start / sampling_rate
            timestamps.append({"start": s_s if return_seconds else speech_start,
                                "end": e_s if return_seconds else len(wav)})
        return timestamps

    return {"model": model, "utils": [get_speech_timestamps]}


registry.register("vad", _load_model)


def analyze(audio: AudioData, threshold: float = THRESHOLD, **kwargs) -> VADResult:
    """
    Batch VAD: detect speech segments in complete audio.
    Returns VADResult with segments and speech fraction.
    """
    import torch

    try:
        bundle = registry.get("vad")
        model = bundle["model"]
        utils = bundle["utils"]
        get_speech_timestamps = utils[0]

        waveform = audio.waveform
        sr = audio.sample_rate

        # Resample to 16kHz if needed
        if sr != SILERO_SR:
            import torchaudio.functional as F
            wv = torch.from_numpy(waveform).unsqueeze(0)
            wv = F.resample(wv, sr, SILERO_SR)
            waveform = wv.squeeze(0).numpy()
            sr = SILERO_SR

        wv_tensor = torch.from_numpy(waveform).float()

        speech_ts = get_speech_timestamps(
            wv_tensor,
            model,
            threshold=threshold,
            sampling_rate=sr,
            min_speech_duration_ms=MIN_SPEECH_MS,
            min_silence_duration_ms=MIN_SILENCE_MS,
            return_seconds=True,
        )

        segments: List[VADSegment] = []
        total_speech = 0.0
        prev_end = 0.0

        for ts in speech_ts:
            start = float(ts["start"])
            end = float(ts["end"])
            if start > prev_end + 0.01:
                segments.append(VADSegment(start_sec=prev_end, end_sec=start, is_speech=False))
            segments.append(VADSegment(start_sec=start, end_sec=end, is_speech=True))
            total_speech += end - start
            prev_end = end

        if prev_end < audio.duration_sec:
            segments.append(VADSegment(start_sec=prev_end, end_sec=audio.duration_sec, is_speech=False))

        ratio = total_speech / max(audio.duration_sec, 1e-6)
        return VADResult(segments=segments, speech_ratio=round(ratio, 4), total_speech_sec=round(total_speech, 3))

    except Exception as e:
        logger.error(f"VAD failed: {e}", exc_info=True)
        return VADResult(success=False, error=str(e))


# --- Streaming helper for real-time use ------------------------------------------

class SileroVADStreamer:
    """
    Stateful streaming VAD for real-time 30ms chunk processing.
    Usage:
        streamer = SileroVADStreamer()
        for chunk in audio_chunks:
            is_speech, prob = streamer.process(chunk)
    """
    def __init__(self, threshold: float = THRESHOLD):
        import torch
        bundle = registry.get("vad")
        self.model = bundle["model"]
        self.threshold = threshold
        self.sr = SILERO_SR
        self.h = torch.zeros(2, 1, 64)
        self.c = torch.zeros(2, 1, 64)

    def process(self, chunk: np.ndarray) -> tuple[bool, float]:
        """Process a single 30ms chunk. Returns (is_speech, probability)."""
        import torch
        x = torch.from_numpy(chunk).float().unsqueeze(0)
        with torch.no_grad():
            prob, self.h, self.c = self.model(x, self.sr, self.h, self.c)
        speech_prob = float(prob.squeeze())
        return speech_prob >= self.threshold, round(speech_prob, 4)

    def reset(self):
        import torch
        self.h = torch.zeros(2, 1, 64)
        self.c = torch.zeros(2, 1, 64)
