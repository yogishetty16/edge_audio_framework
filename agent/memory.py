"""Persistent timeline memory for agentic audio decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schema import AgentDecision


DEFAULT_MEMORY_PATH = Path("agent_memory") / "events.jsonl"


@dataclass
class MemoryEvent:
    """Compact, privacy-aware event stored in the agent timeline."""

    timestamp: str
    event_type: str
    priority: str
    confidence: float
    recommended_action: str
    privacy_mode: str
    risk_signals: List[str] = field(default_factory=list)
    models_used: List[str] = field(default_factory=list)
    transcript: str = ""
    summary: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEvent":
        data = dict(data or {})
        data.setdefault("risk_signals", [])
        data.setdefault("models_used", [])
        data.setdefault("metadata", {})
        data.setdefault("transcript", "")
        data.setdefault("summary", "")
        data.setdefault("source", "")
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: data.get(k) for k in fields})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AudioMemory:
    """JSONL-backed memory for cross-window and cross-run timeline context."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path or DEFAULT_MEMORY_PATH)

    def clear(self) -> None:
        """Delete the memory timeline if it exists."""
        if self.path.exists():
            self.path.unlink()

    def append_decision(self, decision: AgentDecision, source: str = "") -> MemoryEvent:
        """Persist a compact event for a decision."""
        event = self._event_from_decision(decision, source=source)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=True) + "\n")
        return event

    def recent_events(self, window_sec: int = 600, limit: int = 20) -> List[MemoryEvent]:
        """Load recent timeline events within a time window."""
        now = datetime.now(timezone.utc)
        events: List[MemoryEvent] = []

        if not self.path.exists():
            return events

        with self.path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-max(limit * 4, limit):]

        for line in lines:
            try:
                event = MemoryEvent.from_dict(json.loads(line))
                ts = _parse_timestamp(event.timestamp)
                if ts and (now - ts).total_seconds() <= window_sec:
                    events.append(event)
            except Exception:
                continue

        return events[-limit:]

    def build_context(self, decision: AgentDecision, window_sec: int = 600) -> Dict[str, Any]:
        """Build concise context for the current decision from recent events."""
        recent = self.recent_events(window_sec=window_sec)
        related = _related_events(decision, recent)
        risk_history = [
            e for e in related
            if set(decision.risk_signals).intersection(e.risk_signals)
        ]
        repeated_quality = [
            e for e in recent
            if str(e.metadata.get("quality_label", "")).lower() in {"poor", "fair"}
        ]

        escalation_signals: List[str] = []
        if decision.risk_signals and risk_history:
            escalation_signals.append("current risk repeats recent related risk")
        if "low_audio_quality" in decision.risk_signals and len(repeated_quality) >= 2:
            escalation_signals.append("repeated audio quality degradation")
        if decision.priority == "medium" and any(e.priority == "high" for e in related):
            escalation_signals.append("medium event follows recent high-priority event")

        return {
            "recent_count": len(recent),
            "related_count": len(related),
            "related_events": [e.to_dict() for e in related[-5:]],
            "escalation_signals": escalation_signals,
            "window_sec": window_sec,
        }

    def enrich_decision(self, decision: AgentDecision, context: Dict[str, Any]) -> AgentDecision:
        """Use memory context to refine the current decision in place."""
        escalation_signals = context.get("escalation_signals") or []
        if not escalation_signals:
            decision.metadata["memory_recent_events"] = context.get("recent_count", 0)
            return decision

        decision.metadata["memory_recent_events"] = context.get("recent_count", 0)
        decision.metadata["memory_related_events"] = context.get("related_count", 0)
        decision.metadata["memory_escalation_signals"] = escalation_signals
        decision.reasoning.append(
            "recent timeline context increased confidence in the operational state"
        )

        if decision.priority != "high" and decision.risk_signals:
            decision.priority = "high"
            decision.event_type = "possible_safety_incident"
            decision.recommended_action = "create_incident_and_notify_operator"
            decision.confidence = min(0.98, max(decision.confidence, 0.86))
            if "timeline_escalation" not in decision.risk_signals:
                decision.risk_signals.append("timeline_escalation")
            decision.privacy_mode = "metadata_plus_redacted_evidence"
            decision.data_export.update({
                "raw_audio": False,
                "transcript": decision.data_export.get("transcript", False),
                "embeddings": False,
                "metadata": True,
            })

        return decision

    def _event_from_decision(self, decision: AgentDecision, source: str = "") -> MemoryEvent:
        transcript = ""
        if decision.data_export.get("transcript"):
            transcript = str(decision.metadata.get("transcript", "") or "")

        return MemoryEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=decision.event_type,
            priority=decision.priority,
            confidence=decision.confidence,
            recommended_action=decision.recommended_action,
            privacy_mode=decision.privacy_mode,
            risk_signals=list(decision.risk_signals),
            models_used=list(decision.models_used),
            transcript=transcript,
            summary=_summary_from_decision(decision),
            source=source,
            metadata=dict(decision.metadata),
        )


def _summary_from_decision(decision: AgentDecision) -> str:
    reason = decision.reasoning[0] if decision.reasoning else "no reasoning"
    return f"{decision.event_type} / {decision.priority}: {reason}"


def _parse_timestamp(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _related_events(decision: AgentDecision, events: Iterable[MemoryEvent]) -> List[MemoryEvent]:
    current_risks = set(decision.risk_signals)
    related: List[MemoryEvent] = []
    for event in events:
        if event.event_type == decision.event_type:
            related.append(event)
            continue
        if current_risks.intersection(event.risk_signals):
            related.append(event)
            continue
    return related
