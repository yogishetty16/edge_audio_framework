"""Triage agent for adaptive task selection in edge audio intelligence.

The TriageAgent examines recent memory context and policy constraints to
decide which subset of the 14 available tasks should run for the next
audio window.  This keeps compute budgets tight on edge hardware while
ensuring full coverage when risk signals are active.

Classes
-------
TriageAgent
    Selects tasks based on memory context, recent event patterns, and
    policy overrides.
"""

from __future__ import annotations

from typing import Any, Dict, List


class TriageAgent:
    """Selects which tasks to run based on memory context and policy.

    The agent uses a priority-based heuristic:
    1. Always-on tasks run unconditionally.
    2. Recent high-priority events trigger the full task suite.
    3. Quiet periods allow low-cost ambient tasks.
    4. Policy overrides (privacy, power, industrial) constrain the result.
    """

    # Task groups — mirroring the 14-task catalogue
    ALWAYS_ON_TASKS: List[str] = [
        "vad", "impulse_event", "speech_quality", "audio_quality_monitor",
    ]
    SPEECH_TASKS: List[str] = [
        "asr", "emotion", "speaker_id", "keyword_spotting", "accent_lang_id",
    ]
    ENVIRONMENT_TASKS: List[str] = [
        "esc", "acoustic_event_detection", "anomaly_detection",
        "music_speech_detection",
    ]
    FULL_TASKS: List[str] = ["music_genre"]

    def __init__(self) -> None:
        """Initialise the triage agent with task group definitions."""
        pass  # Groups are class-level constants

    def select_tasks(
        self,
        memory_context: dict,
        policy: str = "balanced",
    ) -> List[str]:
        """Choose which tasks to execute for the next audio window.

        Parameters
        ----------
        memory_context : dict
            Recent events from the watchdog / memory store.  Expected
            keys: ``recent_events`` (list of event dicts), optionally
            ``watchdog_report`` and ``feedback_for_triage``.
        policy : str
            One of ``balanced``, ``privacy_first``, ``low_power``,
            ``industrial``.

        Returns
        -------
        list[str]
            Deduplicated, policy-constrained task list.
        """
        try:
            memory_context = memory_context or {}
            selected: List[str] = list(self.ALWAYS_ON_TASKS)

            recent_events = memory_context.get("recent_events", [])
            if not recent_events:
                recent_events = memory_context.get("related_events", [])

            last_5 = recent_events[-5:] if recent_events else []
            last_3 = recent_events[-3:] if recent_events else []

            # Check for watchdog feedback overrides
            feedback = memory_context.get("feedback_for_triage", {})
            if feedback:
                if feedback.get("force_quality_check"):
                    selected.append("audio_quality_monitor")
                if feedback.get("skip_speech_tasks"):
                    # Only add environment tasks when no speech at all
                    selected.extend(self.ENVIRONMENT_TASKS)
                    return self._apply_policy(selected, policy)

            # Rule 1: Any recent HIGH or risk signals → full suite
            has_high = any(
                _event_priority(e) == "high"
                or "impulse_event" in _event_risks(e)
                or "alert_sound" in _event_risks(e)
                for e in last_5
            )
            if has_high:
                selected.extend(self.SPEECH_TASKS)
                selected.extend(self.ENVIRONMENT_TASKS)
                return self._apply_policy(selected, policy)

            # Rule 2: Repeated poor quality → anomaly only
            poor_count = sum(
                1 for e in last_5
                if _event_quality(e) in ("poor", "fair")
            )
            if poor_count >= 3:
                selected.append("anomaly_detection")
                return self._apply_policy(selected, policy)

            # Rule 3: No speech in last 3 events → skip speech, add env
            has_speech = any(
                _event_type(e) in (
                    "speech_detected", "speech_workflow_event",
                    "possible_safety_incident",
                )
                for e in last_3
            )
            if not has_speech and last_3:
                selected.extend(self.ENVIRONMENT_TASKS)
                return self._apply_policy(selected, policy)

            # Rule 4: All normal in last 3 → add full tasks
            all_normal = all(
                _event_type(e) == "normal_audio" for e in last_3
            ) if last_3 else False
            if all_normal:
                selected.extend(self.FULL_TASKS)

            # Default: add both speech + environment
            selected.extend(self.SPEECH_TASKS)
            selected.extend(self.ENVIRONMENT_TASKS)

            return self._apply_policy(selected, policy)

        except Exception:
            # Safe fallback — always-on tasks only
            return list(self.ALWAYS_ON_TASKS)

    def explain_selection(
        self,
        selected_tasks: list,
        memory_context: dict,
    ) -> str:
        """Return a one-sentence explanation for the task selection.

        Parameters
        ----------
        selected_tasks : list
            Tasks selected by ``select_tasks``.
        memory_context : dict
            Same memory context passed to ``select_tasks``.

        Returns
        -------
        str
            Human-readable explanation of the selection rationale.
        """
        try:
            memory_context = memory_context or {}
            recent = memory_context.get("recent_events", [])
            if not recent:
                recent = memory_context.get("related_events", [])
            last_5 = recent[-5:] if recent else []

            has_high = any(
                _event_priority(e) == "high" for e in last_5
            )
            if has_high:
                return (
                    "Recent HIGH priority event detected — running full task "
                    "suite for maximum situational awareness."
                )

            poor_count = sum(
                1 for e in last_5
                if _event_quality(e) in ("poor", "fair")
            )
            if poor_count >= 3:
                return (
                    "Repeated poor audio quality detected — focusing on "
                    "anomaly detection while skipping speech tasks."
                )

            n = len(selected_tasks)
            if n <= len(self.ALWAYS_ON_TASKS) + 1:
                return (
                    "Quiet period with no recent risk signals — running "
                    "lightweight always-on tasks to conserve compute."
                )

            return (
                f"Standard triage selected {n} tasks based on current "
                f"memory context and policy constraints."
            )

        except Exception:
            return "Task selection completed with default heuristics."

    # ── internal helpers ─────────────────────────────────────────────

    def _apply_policy(self, tasks: List[str], policy: str) -> List[str]:
        """Apply policy overrides and deduplicate the task list."""
        try:
            result = _dedupe(tasks)

            if policy == "privacy_first":
                result = [t for t in result if t not in ("speaker_id", "accent_lang_id")]

            if policy == "low_power":
                result = result[:6]

            if policy == "industrial":
                for forced in ("esc", "anomaly_detection"):
                    if forced not in result:
                        result.append(forced)

            return _dedupe(result)

        except Exception:
            return list(self.ALWAYS_ON_TASKS)


# ── private helpers ──────────────────────────────────────────────────

def _dedupe(items: list) -> list:
    """Return a deduplicated list preserving insertion order."""
    seen: set = set()
    out: list = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _event_priority(event: dict) -> str:
    """Extract priority from an event dict, defaulting to 'low'."""
    if isinstance(event, dict):
        return str(event.get("priority", "low")).lower()
    return "low"


def _event_risks(event: dict) -> list:
    """Extract risk signals from an event dict."""
    if isinstance(event, dict):
        return list(event.get("risk_signals", []))
    return []


def _event_type(event: dict) -> str:
    """Extract event_type from an event dict."""
    if isinstance(event, dict):
        return str(event.get("event_type", ""))
    return ""


def _event_quality(event: dict) -> str:
    """Extract quality_label from event metadata."""
    if isinstance(event, dict):
        meta = event.get("metadata", {})
        if isinstance(meta, dict):
            return str(meta.get("quality_label", "")).lower()
    return ""
