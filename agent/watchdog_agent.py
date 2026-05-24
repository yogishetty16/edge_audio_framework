"""Watchdog agent for long-term trend analysis and hardware health.

The WatchdogAgent monitors the stream of agent decisions over time,
detecting trends in anomaly rates, audio quality degradation, and risk
frequency.  It produces periodic watchdog reports and provides feedback
to the TriageAgent for adaptive task selection.

Classes
-------
WatchdogAgent
    Persistent event recorder with periodic trend analysis.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class WatchdogAgent:
    """Record events and periodically analyse trends across the session.

    Every ``WATCHDOG_INTERVAL`` events, the agent runs a full analysis
    over the last 50 events and produces a watchdog report covering:
    - trend analysis (anomaly, quality, risk)
    - hardware health (microphone degradation)
    - shift summary (event counts and top signals)
    - feedback for the triage agent (sensitivity / task suggestions)
    """

    WATCHDOG_INTERVAL: int = 10

    def __init__(self, memory_path: str = "agent_memory/events.jsonl") -> None:
        """Initialise the watchdog with a JSONL memory path.

        Parameters
        ----------
        memory_path : str
            Path to the persistent JSONL event store.
        """
        self.memory_path = Path(memory_path)
        self._event_count: int = 0
        self._latest_report: Optional[dict] = None

    # ── event recording ──────────────────────────────────────────────

    def record_event(self, event: dict) -> Optional[dict]:
        """Persist an event and run analysis every ``WATCHDOG_INTERVAL`` events.

        Parameters
        ----------
        event : dict
            Agent decision dict to persist.

        Returns
        -------
        dict or None
            A watchdog report dict if the interval was reached, else None.
        """
        try:
            self._event_count += 1
            self._append_jsonl(event)

            if self._event_count % self.WATCHDOG_INTERVAL == 0:
                report = self.analyze()
                self._latest_report = report
                return report
            return None
        except Exception:
            return None

    # ── trend analysis ───────────────────────────────────────────────

    def analyze(self) -> dict:
        """Analyse the last 50 events and produce a comprehensive report.

        Returns
        -------
        dict
            Watchdog report with keys: ``trend_analysis``,
            ``hardware_health``, ``shift_summary``, ``feedback_for_triage``.
        """
        try:
            events = self._read_last_n(50)
            return {
                "trend_analysis": self._trend_analysis(events),
                "hardware_health": self._hardware_health(events),
                "shift_summary": self._shift_summary(events),
                "feedback_for_triage": self._feedback_for_triage(events),
            }
        except Exception:
            return {
                "trend_analysis": {"anomaly_trend": "stable",
                                    "quality_trend": "stable",
                                    "risk_frequency": 0.0,
                                    "dominant_event_type": "unknown"},
                "hardware_health": {"status": "healthy",
                                     "consecutive_poor_quality": 0,
                                     "avg_snr_db": 0.0,
                                     "recommendation": "Insufficient data for assessment"},
                "shift_summary": {"total_events": 0,
                                   "high_priority_count": 0,
                                   "medium_priority_count": 0,
                                   "low_priority_count": 0,
                                   "top_risk_signals": [],
                                   "top_event_types": [],
                                   "summary_text": "No events to summarise."},
                "feedback_for_triage": {"elevate_sensitivity": False,
                                         "skip_speech_tasks": False,
                                         "force_quality_check": False,
                                         "context_note": "Watchdog analysis unavailable."},
            }

    # ── memory context ───────────────────────────────────────────────

    def get_memory_context(self, n: int = 5) -> dict:
        """Return the last *n* events as a dict for TriageAgent / SynthesisAgent.

        Parameters
        ----------
        n : int
            Number of recent events to include.

        Returns
        -------
        dict
            Dict with ``recent_events`` (list) and ``watchdog_report``
            (dict or None).
        """
        try:
            events = self._read_last_n(n)
            return {
                "recent_events": events,
                "watchdog_report": self._latest_report,
            }
        except Exception:
            return {
                "recent_events": [],
                "watchdog_report": None,
            }

    # ── internal: trend analysis ─────────────────────────────────────

    def _trend_analysis(self, events: List[dict]) -> dict:
        """Compute anomaly, quality, and risk trends."""
        try:
            n = len(events)
            if n < 2:
                return {
                    "anomaly_trend": "stable",
                    "quality_trend": "stable",
                    "risk_frequency": 0.0,
                    "dominant_event_type": _dominant(events, "event_type"),
                }

            # Split into recent-10 vs previous-10
            recent = events[-10:] if n >= 10 else events[-(n // 2):]
            previous = events[-20:-10] if n >= 20 else events[:len(recent)]

            # Anomaly trend
            recent_anomaly_avg = _avg_meta(recent, "anomaly_score")
            prev_anomaly_avg = _avg_meta(previous, "anomaly_score")
            if recent_anomaly_avg > prev_anomaly_avg + 0.05:
                anomaly_trend = "rising"
            elif recent_anomaly_avg < prev_anomaly_avg - 0.05:
                anomaly_trend = "falling"
            else:
                anomaly_trend = "stable"

            # Quality trend
            recent_poor = sum(1 for e in recent if _quality(e) in ("poor", "fair"))
            prev_poor = sum(1 for e in previous if _quality(e) in ("poor", "fair"))
            if recent_poor > prev_poor + 1:
                quality_trend = "degrading"
            elif recent_poor < prev_poor - 1:
                quality_trend = "improving"
            else:
                quality_trend = "stable"

            # Risk frequency
            high_count = sum(1 for e in events if _priority(e) == "high")
            risk_frequency = round(high_count / max(n, 1), 4)

            return {
                "anomaly_trend": anomaly_trend,
                "quality_trend": quality_trend,
                "risk_frequency": risk_frequency,
                "dominant_event_type": _dominant(events, "event_type"),
            }
        except Exception:
            return {
                "anomaly_trend": "stable", "quality_trend": "stable",
                "risk_frequency": 0.0, "dominant_event_type": "unknown",
            }

    # ── internal: hardware health ────────────────────────────────────

    def _hardware_health(self, events: List[dict]) -> dict:
        """Assess microphone / hardware health from quality signals."""
        try:
            if not events:
                return {
                    "status": "healthy",
                    "consecutive_poor_quality": 0,
                    "avg_snr_db": 0.0,
                    "recommendation": "No events available for assessment.",
                }

            # Count consecutive poor quality from the end
            consecutive = 0
            for e in reversed(events):
                if _quality(e) in ("poor", "fair"):
                    consecutive += 1
                else:
                    break

            # Average SNR
            snr_values = [_snr(e) for e in events if _snr(e) > 0]
            avg_snr = round(sum(snr_values) / max(len(snr_values), 1), 2)

            if consecutive >= 5 or avg_snr < 5:
                status = "critical"
                rec = (
                    "Microphone showing critical degradation — "
                    "immediate maintenance required."
                )
            elif consecutive >= 3 or avg_snr < 10:
                status = "degrading"
                rec = (
                    "Microphone showing signs of degradation — "
                    "schedule maintenance."
                )
            else:
                status = "healthy"
                rec = "Hardware operating within normal parameters."

            return {
                "status": status,
                "consecutive_poor_quality": consecutive,
                "avg_snr_db": avg_snr,
                "recommendation": rec,
            }
        except Exception:
            return {
                "status": "healthy", "consecutive_poor_quality": 0,
                "avg_snr_db": 0.0,
                "recommendation": "Health check encountered an error.",
            }

    # ── internal: shift summary ──────────────────────────────────────

    def _shift_summary(self, events: List[dict]) -> dict:
        """Produce a human-readable shift/session summary."""
        try:
            n = len(events)
            high = sum(1 for e in events if _priority(e) == "high")
            medium = sum(1 for e in events if _priority(e) == "medium")
            low = n - high - medium

            # Top risk signals
            all_risks: List[str] = []
            for e in events:
                all_risks.extend(e.get("risk_signals", []))
            top_risks = [s for s, _ in Counter(all_risks).most_common(3)]

            # Top event types
            all_types = [e.get("event_type", "") for e in events]
            top_types = [t for t, _ in Counter(all_types).most_common(3)]

            # Summary text
            if high > 0:
                summary = (
                    f"Session processed {n} audio windows with {high} high-priority "
                    f"event(s) requiring operator attention. "
                    f"The most common event type was '{top_types[0] if top_types else 'unknown'}'. "
                    f"Continued monitoring is recommended."
                )
            elif medium > 0:
                summary = (
                    f"Session processed {n} audio windows with {medium} medium-priority "
                    f"event(s). No critical incidents were detected. "
                    f"Routine review of flagged events is suggested."
                )
            else:
                summary = (
                    f"Session processed {n} audio windows with no elevated-priority "
                    f"events. The environment appears stable and normal."
                )

            return {
                "total_events": n,
                "high_priority_count": high,
                "medium_priority_count": medium,
                "low_priority_count": low,
                "top_risk_signals": top_risks,
                "top_event_types": top_types,
                "summary_text": summary,
            }
        except Exception:
            return {
                "total_events": 0, "high_priority_count": 0,
                "medium_priority_count": 0, "low_priority_count": 0,
                "top_risk_signals": [], "top_event_types": [],
                "summary_text": "Summary generation encountered an error.",
            }

    # ── internal: feedback for triage ────────────────────────────────

    def _feedback_for_triage(self, events: List[dict]) -> dict:
        """Generate adaptive feedback for the TriageAgent."""
        try:
            n = len(events)
            high_count = sum(1 for e in events if _priority(e) == "high")
            risk_freq = high_count / max(n, 1)

            last_10 = events[-10:] if len(events) >= 10 else events
            speech_in_last_10 = any(
                e.get("event_type", "") in (
                    "speech_detected", "speech_workflow_event",
                    "possible_safety_incident",
                )
                for e in last_10
            )

            quality_degrading = sum(
                1 for e in last_10 if _quality(e) in ("poor", "fair")
            ) >= 3

            notes: List[str] = []
            elevate = risk_freq > 0.3
            skip_speech = not speech_in_last_10
            force_quality = quality_degrading

            if elevate:
                notes.append("High risk frequency detected — elevating sensitivity")
            if skip_speech:
                notes.append("No speech in recent events — speech tasks can be skipped")
            if force_quality:
                notes.append("Quality degradation detected — forcing quality checks")

            return {
                "elevate_sensitivity": elevate,
                "skip_speech_tasks": skip_speech,
                "force_quality_check": force_quality,
                "context_note": "; ".join(notes) if notes else "No adjustments needed.",
            }
        except Exception:
            return {
                "elevate_sensitivity": False,
                "skip_speech_tasks": False,
                "force_quality_check": False,
                "context_note": "Feedback generation encountered an error.",
            }

    # ── JSONL I/O ────────────────────────────────────────────────────

    def _append_jsonl(self, event: dict) -> None:
        """Append one event to the JSONL memory file."""
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            # Add timestamp if missing
            record = dict(event)
            if "timestamp" not in record:
                record["timestamp"] = datetime.now(timezone.utc).isoformat()
            with self.memory_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
        except Exception:
            pass  # Never crash the pipeline

    def _read_last_n(self, n: int) -> List[dict]:
        """Read the last *n* events from the JSONL file."""
        try:
            if not self.memory_path.exists():
                return []
            with self.memory_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            events: List[dict] = []
            for line in lines[-(n * 2):]:
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
            return events[-n:]
        except Exception:
            return []


# ── private helpers ──────────────────────────────────────────────────

def _priority(event: dict) -> str:
    """Extract priority from an event dict."""
    return str(event.get("priority", "low")).lower() if isinstance(event, dict) else "low"


def _quality(event: dict) -> str:
    """Extract quality label from event metadata."""
    if isinstance(event, dict):
        meta = event.get("metadata", {})
        if isinstance(meta, dict):
            return str(meta.get("quality_label", "")).lower()
    return ""


def _snr(event: dict) -> float:
    """Extract SNR dB from event metadata."""
    if isinstance(event, dict):
        meta = event.get("metadata", {})
        if isinstance(meta, dict):
            try:
                return float(meta.get("snr_db", 0.0))
            except (TypeError, ValueError):
                pass
    return 0.0


def _avg_meta(events: List[dict], key: str) -> float:
    """Compute average of a metadata field across events."""
    values = []
    for e in events:
        meta = e.get("metadata", {}) if isinstance(e, dict) else {}
        if isinstance(meta, dict):
            try:
                v = float(meta.get(key, 0.0))
                values.append(v)
            except (TypeError, ValueError):
                pass
    return round(sum(values) / max(len(values), 1), 4) if values else 0.0


def _dominant(events: List[dict], key: str) -> str:
    """Find the most common value for *key* across events."""
    if not events:
        return "unknown"
    counter = Counter(
        str(e.get(key, "unknown")) for e in events if isinstance(e, dict)
    )
    most = counter.most_common(1)
    return most[0][0] if most else "unknown"
