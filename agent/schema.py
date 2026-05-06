"""Typed contracts for the agentic audio orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass(frozen=True)
class AgentPolicy:
    """Runtime policy used by the agent to balance quality, cost, and privacy."""

    name: str = "balanced"
    always_on_tasks: List[str] = field(default_factory=list)
    speech_follow_up_tasks: List[str] = field(default_factory=list)
    non_speech_follow_up_tasks: List[str] = field(default_factory=list)
    high_risk_follow_up_tasks: List[str] = field(default_factory=list)
    strict_privacy: bool = True
    allow_raw_audio_export: bool = False
    max_parallel_tasks: int = 6
    min_speech_ratio: float = 0.05
    low_quality_snr_db: float = 8.0
    low_quality_labels: List[str] = field(default_factory=lambda: ["poor"])
    stress_emotions: List[str] = field(
        default_factory=lambda: ["angry", "fear", "fearful", "sad", "disgust"]
    )
    alert_keywords: List[str] = field(
        default_factory=lambda: [
            "help",
            "stop",
            "emergency",
            "danger",
            "fire",
            "alarm",
            "evacuate",
            "shutdown",
            "leak",
            "injured",
            "pain",
        ]
    )
    stress_score_threshold: float = 0.45
    esc_alert_score_threshold: float = 0.25
    keyword_alert_score_threshold: float = 0.50
    anomaly_score_threshold: float = -0.10
    low_confidence_threshold: float = 0.55


@dataclass
class TaskPlan:
    """A plan for the next task batch."""

    tasks: List[str]
    reason: str
    skipped_tasks: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDecision:
    """Final agent output generated from specialist model results."""

    event_type: str
    priority: str
    confidence: float
    recommended_action: str
    privacy_mode: str
    models_used: List[str]
    reasoning: List[str]
    follow_up_tasks: List[str] = field(default_factory=list)
    risk_signals: List[str] = field(default_factory=list)
    data_export: Dict[str, bool] = field(default_factory=dict)
    task_health: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
