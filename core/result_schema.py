"""
core/result_schema.py
=====================
Dataclasses for every task's output.
All inherit BaseResult which provides success/error fields and dict conversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class BaseResult:
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # Allow dict-style .get() access for pipeline compatibility
    def get(self, key: str, default=None):
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str):
        return self.to_dict()[key]


# ── VAD ────────────────────────────────────────────────────────────────────────

@dataclass
class VADSegment:
    start_sec: float = 0.0
    end_sec: float = 0.0
    is_speech: bool = False


@dataclass
class VADResult(BaseResult):
    speech_ratio: float = 0.0
    total_speech_sec: float = 0.0
    is_speech: bool = False
    segments: List[Dict] = field(default_factory=list)


# ── ASR ────────────────────────────────────────────────────────────────────────

@dataclass
class ASRResult(BaseResult):
    text: str = ""
    language: str = ""
    language_probability: float = 0.0
    segments: List[Dict] = field(default_factory=list)


# ── Keyword Spotting ────────────────────────────────────────────────────────────

@dataclass
class KeywordResult(BaseResult):
    detected_keywords: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    top_label: str = ""
    top_score: float = 0.0


# ── Speaker ID ─────────────────────────────────────────────────────────────────

@dataclass
class SpeakerSegment:
    start_sec: float = 0.0
    end_sec: float = 0.0
    speaker_id: str = ""


@dataclass
class SpeakerIDResult(BaseResult):
    segments: List[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0
    embedding: Optional[List[float]] = None


@dataclass
class SpeakerVerificationResult(BaseResult):
    score: float = 0.0
    is_same_speaker: bool = False
    threshold: float = 0.25


# ── Emotion ────────────────────────────────────────────────────────────────────

@dataclass
class EmotionResult(BaseResult):
    top_emotion: str = ""
    top_score: float = 0.0
    all_scores: Dict[str, float] = field(default_factory=dict)
    sentiment: Optional[str] = None


# ── Speech Quality ─────────────────────────────────────────────────────────────

@dataclass
class SpeechQualityResult(BaseResult):
    dnsmos_overall: float = 0.0
    dnsmos_signal: float = 0.0
    dnsmos_background: float = 0.0
    dnsmos_noisy: float = 0.0
    pesq_score: Optional[float] = None
    stoi_score: Optional[float] = None
    snr_db: float = 0.0


# ── Language / Accent ID ───────────────────────────────────────────────────────

@dataclass
class LanguageIDResult(BaseResult):
    top_language: str = ""
    top_language_score: float = 0.0
    all_language_scores: Dict[str, float] = field(default_factory=dict)
    top_accent: Optional[str] = None
    top_accent_score: float = 0.0
    all_accent_scores: Dict[str, float] = field(default_factory=dict)


# ── ESC ────────────────────────────────────────────────────────────────────────

@dataclass
class ESCResult(BaseResult):
    top_class: str = ""
    top_score: float = 0.0
    all_scores: Dict[str, float] = field(default_factory=dict)


# ── Acoustic Event Detection ───────────────────────────────────────────────────

@dataclass
class AcousticEvent:
    label: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
    score: float = 0.0


@dataclass
class AEDResult(BaseResult):
    events: List[AcousticEvent] = field(default_factory=list)
    frame_labels: List[str] = field(default_factory=list)


# ── Audio Quality Monitor ──────────────────────────────────────────────────────

@dataclass
class AudioQualityResult(BaseResult):
    mos: float = 0.0
    snr_db: float = 0.0
    clipping_detected: bool = False
    dc_offset: float = 0.0
    rms_db: float = -100.0
    silence_fraction: float = 0.0
    quality_label: str = "unknown"


# ── Music / Speech Detection ───────────────────────────────────────────────────

@dataclass
class MusicSpeechSegment:
    start_sec: float = 0.0
    end_sec: float = 0.0
    label: str = ""


@dataclass
class MusicSpeechResult(BaseResult):
    segments: List[MusicSpeechSegment] = field(default_factory=list)
    speech_fraction: float = 0.0
    music_fraction: float = 0.0
    noise_fraction: float = 0.0


# ── Anomaly Detection ──────────────────────────────────────────────────────────

@dataclass
class AnomalyResult(BaseResult):
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    threshold: float = -0.1
    anomalous_segments: List[Dict] = field(default_factory=list)
