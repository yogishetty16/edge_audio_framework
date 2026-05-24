"""Agentic edge audio orchestrator — 3-agent reasoning system.

This module replaces the original rule-based ``AudioAgent`` with a
proper 3-agent architecture (Triage → Synthesis → Watchdog) while
keeping the original rule engine as a fallback.  It is a **drop-in
replacement**: same class name, same constructor signature, and same
public methods so that ``fast_run.py`` works unchanged.

Architecture
------------
1. **TriageAgent** — selects which tasks to run based on recent memory
   and policy constraints.
2. **SynthesisAgent** — consumes task outputs and produces a complete
   ``AgentDecision`` with chain-of-thought reasoning.
3. **WatchdogAgent** — records events, detects trends, and feeds back
   to the triage layer.

The original ``audio_agent_rules.py`` is imported as the fallback: if
any agent raises an exception, the pipeline degrades gracefully to the
proven rule engine.

Classes
-------
AudioAgent
    Drop-in replacement orchestrator for the edge audio framework.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Iterable, List, Optional

from .schema import AgentDecision, AgentPolicy, TaskPlan
from .policy import get_policy

# ── agents ───────────────────────────────────────────────────────────
from .triage_agent import TriageAgent
from .synthesis_agent import SynthesisAgent
from .watchdog_agent import WatchdogAgent

# ── fallback rule engine ─────────────────────────────────────────────
from . import audio_agent_rules as _rules

logger = logging.getLogger(__name__)


class AudioAgent:
    """Drop-in orchestrator with a 3-agent reasoning system.

    This class exposes the same public API as the original
    ``AudioAgent`` in ``audio_agent_rules.py``:

    * ``__init__(profile, policy)``
    * ``plan_initial_tasks(requested_tasks)``
    * ``plan_follow_up_tasks(results, requested_tasks, already_run)``
    * ``decide(results)``

    Internally it delegates to TriageAgent, SynthesisAgent, and
    WatchdogAgent, falling back to the rule engine on any failure.
    """

    def __init__(
        self,
        profile: str = "balanced",
        policy: Optional[AgentPolicy] = None,
    ) -> None:
        """Initialise the 3-agent system and the fallback rule engine.

        Parameters
        ----------
        profile : str
            Policy profile name (``balanced``, ``industrial_safety``,
            ``privacy_first``, ``low_power``).
        policy : AgentPolicy or None
            Optional explicit policy object.  If None, resolved from
            *profile*.
        """
        # Core policy — shared with the fallback engine
        self.policy = policy or get_policy(profile)

        # Fallback rule engine (same constructor signature)
        self._fallback = _rules.AudioAgent(profile=profile, policy=self.policy)

        # 3-agent system
        self._triage = TriageAgent()
        self._synthesis = SynthesisAgent()
        self._watchdog = WatchdogAgent()

        # Track the active policy name for the synthesis agent
        self._policy_name = self.policy.name

    # ── task planning (delegates to fallback for compatibility) ───────

    def plan_initial_tasks(
        self,
        requested_tasks: Optional[Iterable[str]] = None,
    ) -> TaskPlan:
        """Return the first lightweight task batch.

        Delegates to the original rule engine to maintain exact
        compatibility with ``fast_run.py``'s orchestrated mode.
        """
        try:
            return self._fallback.plan_initial_tasks(requested_tasks)
        except Exception as exc:
            logger.warning("plan_initial_tasks fallback failed: %s", exc)
            return TaskPlan(tasks=list(requested_tasks or []), reason="Fallback planning.")

    def plan_follow_up_tasks(
        self,
        results: Dict[str, Dict[str, Any]],
        requested_tasks: Optional[Iterable[str]] = None,
        already_run: Optional[Iterable[str]] = None,
    ) -> TaskPlan:
        """Plan the next task batch from first-pass results.

        Delegates to the original rule engine to maintain exact
        compatibility with ``fast_run.py``'s orchestrated mode.
        """
        try:
            return self._fallback.plan_follow_up_tasks(
                results, requested_tasks=requested_tasks, already_run=already_run
            )
        except Exception as exc:
            logger.warning("plan_follow_up_tasks fallback failed: %s", exc)
            return TaskPlan(tasks=[], reason="Fallback follow-up planning.")

    # ── task selection via triage agent ───────────────────────────────

    def get_tasks(self, memory_context: dict = None) -> List[str]:
        """Select tasks using the TriageAgent.

        Falls back to the existing policy logic in
        ``audio_agent_rules.py`` if the triage agent raises any
        exception.

        Parameters
        ----------
        memory_context : dict or None
            Recent memory context.  If None, fetched from WatchdogAgent.

        Returns
        -------
        list[str]
            Selected task names.
        """
        try:
            if memory_context is None:
                memory_context = self._watchdog.get_memory_context()
            return self._triage.select_tasks(memory_context, self._policy_name)
        except Exception as exc:
            logger.warning("TriageAgent failed, falling back to rules: %s", exc)
            try:
                return list(self.policy.always_on_tasks)
            except Exception:
                return ["vad", "audio_quality_monitoring", "anomaly_detection", "impulse_event"]

    # ── main decision pipeline ───────────────────────────────────────

    def decide(self, results: Dict[str, Dict[str, Any]]) -> AgentDecision:
        """Create one actionable decision from task outputs.

        Pipeline
        --------
        1. Fetch memory context from WatchdogAgent.
        2. Run SynthesisAgent to produce a full decision dict.
        3. On ANY exception, fall back to ``audio_agent_rules.py``.
        4. Record the event in WatchdogAgent.
        5. If a watchdog report is returned, attach it.
        6. Return the final ``AgentDecision``.

        Parameters
        ----------
        results : dict
            Raw task outputs keyed by task name.

        Returns
        -------
        AgentDecision
            Dataclass instance with all schema fields populated.
        """
        # Step 1: memory context
        try:
            memory_context = self._watchdog.get_memory_context()
        except Exception:
            memory_context = {"recent_events": [], "watchdog_report": None}

        # Step 2: try the 3-agent system
        decision_dict = None
        used_fallback = False
        try:
            # Inject triage explanation into memory context
            try:
                triage_tasks = self._triage.select_tasks(
                    memory_context, self._policy_name
                )
                triage_explanation = self._triage.explain_selection(
                    triage_tasks, memory_context
                )
                memory_context["triage_explanation"] = triage_explanation
            except Exception:
                memory_context["triage_explanation"] = "Triage explanation unavailable."

            decision_dict = self._synthesis.synthesize(
                results, memory_context, self._policy_name
            )

        except Exception as exc:
            # Step 3: fallback to rule engine
            logger.warning(
                "SynthesisAgent failed, falling back to rules: %s", exc
            )
            used_fallback = True

        if decision_dict is None or used_fallback:
            try:
                fallback_decision = self._fallback.decide(results or {})
                decision_dict = fallback_decision.to_dict()
                # Augment with new fields
                decision_dict.setdefault("incident_report", "Decision produced by fallback rule engine.")
                decision_dict.setdefault("triage_explanation", "Fallback mode — triage agent was bypassed.")
                decision_dict.setdefault("watchdog_report", None)
                decision_dict["reasoning"] = decision_dict.get("reasoning", []) + [
                    f"[fallback] Rule engine was used due to synthesis failure"
                ]
            except Exception as fallback_exc:
                logger.error("Fallback rule engine also failed: %s", fallback_exc)
                decision_dict = {
                    "event_type": "normal_audio",
                    "priority": "low",
                    "confidence": 0.10,
                    "recommended_action": "ignore_or_continue_monitoring",
                    "privacy_mode": "metadata_only",
                    "models_used": sorted((results or {}).keys()),
                    "reasoning": [f"Both agents failed: {fallback_exc}"],
                    "follow_up_tasks": [],
                    "risk_signals": [],
                    "data_export": {"raw_audio": False, "transcript": False,
                                    "embeddings": False, "metadata": True},
                    "task_health": {},
                    "metadata": {"policy": self._policy_name},
                    "incident_report": "System error — both agents failed.",
                    "triage_explanation": "Unavailable due to system error.",
                    "watchdog_report": None,
                }

        # Step 4: record event in watchdog
        watchdog_report = None
        try:
            watchdog_report = self._watchdog.record_event(decision_dict)
        except Exception:
            pass

        # Step 5: attach watchdog report if generated
        if watchdog_report is not None:
            decision_dict["watchdog_report"] = watchdog_report

        # Step 6: convert to AgentDecision dataclass
        return _dict_to_decision(decision_dict)


# ── helpers ──────────────────────────────────────────────────────────

def _dict_to_decision(d: dict) -> AgentDecision:
    """Convert a decision dict to an AgentDecision dataclass.

    Extra keys (``incident_report``, ``triage_explanation``,
    ``watchdog_report``) are placed in ``metadata`` so that
    ``to_dict()`` exposes them.
    """
    try:
        # Core fields
        decision = AgentDecision(
            event_type=d.get("event_type", "normal_audio"),
            priority=d.get("priority", "low"),
            confidence=d.get("confidence", 0.10),
            recommended_action=d.get("recommended_action", "ignore_or_continue_monitoring"),
            privacy_mode=d.get("privacy_mode", "metadata_only"),
            models_used=d.get("models_used", []),
            reasoning=d.get("reasoning", []),
            follow_up_tasks=d.get("follow_up_tasks", []),
            risk_signals=d.get("risk_signals", []),
            data_export=d.get("data_export", {}),
            task_health=d.get("task_health", {}),
            metadata=d.get("metadata", {}),
        )

        # Attach new fields into metadata so to_dict() exposes them
        decision.metadata["incident_report"] = d.get(
            "incident_report", ""
        )
        decision.metadata["triage_explanation"] = d.get(
            "triage_explanation", ""
        )
        decision.metadata["watchdog_report"] = d.get(
            "watchdog_report", None
        )

        return decision
    except Exception:
        return AgentDecision(
            event_type="normal_audio",
            priority="low",
            confidence=0.10,
            recommended_action="ignore_or_continue_monitoring",
            privacy_mode="metadata_only",
            models_used=[],
            reasoning=["Failed to construct AgentDecision"],
        )


# ── smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json

    print("=" * 60)
    print("  SMOKE TEST -- 3-Agent Audio Intelligence System")
    print("=" * 60)

    # Dummy task results simulating a moderate-risk scenario
    dummy_results = {
        "vad": {
            "is_speech": True,
            "speech_ratio": 0.94,
            "total_speech_sec": 4.7,
            "success": True,
        },
        "asr": {
            "text": "Help me, there is a fire in the building!",
            "language": "en",
            "success": True,
        },
        "emotion": {
            "top_emotion": "angry",
            "top_score": 0.81,
            "success": True,
        },
        "keyword_spotting": {
            "top_label": "help",
            "top_score": 0.88,
            "success": True,
        },
        "esc": {
            "top_class": "scream",
            "top_score": 0.74,
            "success": True,
        },
        "anomaly_detection": {
            "is_anomaly": False,
            "anomaly_score": 0.12,
            "success": True,
        },
        "impulse_event": {
            "is_impulse": False,
            "event_label": "none",
            "confidence": 0.05,
            "success": True,
        },
        "audio_quality_monitoring": {
            "quality_label": "good",
            "snr_db": 22.5,
            "mos": 3.8,
            "clipping_detected": False,
            "success": True,
        },
        "speaker_id": {
            "num_speakers": 2,
            "success": True,
        },
        "music_speech_detection": {
            "speech_fraction": 0.85,
            "music_fraction": 0.10,
            "success": True,
        },
    }

    agent = AudioAgent(profile="balanced")

    print("\n[1] Testing get_tasks()...")
    tasks = agent.get_tasks()
    print(f"    Selected tasks: {tasks}")

    print("\n[2] Testing decide() with dummy results...")
    decision = agent.decide(dummy_results)
    payload = decision.to_dict()

    print(f"\n{'=' * 60}")
    print("  FULL AGENT DECISION (JSON)")
    print(f"{'=' * 60}")
    print(_json.dumps(payload, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Event type  : {payload['event_type']}")
    print(f"  Priority    : {payload['priority'].upper()}")
    print(f"  Confidence  : {payload['confidence'] * 100:.0f}%")
    print(f"  Action      : {payload['recommended_action']}")
    print(f"  Risk signals: {', '.join(payload['risk_signals']) or 'none'}")

    report = payload.get("metadata", {}).get("incident_report", "")
    if report:
        print(f"\n  Incident report:")
        print(f"    {report}")

    reasoning = payload.get("reasoning", [])
    if reasoning:
        print(f"\n  Reasoning chain ({len(reasoning)} steps):")
        for i, step in enumerate(reasoning[:6], 1):
            print(f"    {i}. {step}")
        if len(reasoning) > 6:
            print(f"    ... and {len(reasoning) - 6} more step(s)")

    print(f"\n{'=' * 60}")
    print("  SMOKE TEST PASSED [OK]")
    print(f"{'=' * 60}")
