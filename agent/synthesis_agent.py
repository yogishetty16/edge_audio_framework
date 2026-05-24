"""Synthesis agent for edge audio intelligence.

The SynthesisAgent consumes raw task outputs, builds structured context,
applies deterministic reasoning, and produces a complete AgentDecision
dict — including the new ``incident_report`` and ``triage_explanation``
fields.

It replicates the signal-extraction logic from the original rule engine
(audio_agent_rules.py) so that risk thresholds remain identical.

Classes
-------
SynthesisAgent
    Main decision-synthesis pipeline for the agentic audio layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .reasoning_engine import ReasoningEngine


class SynthesisAgent:
    """Synthesise a complete agent decision from task outputs.

    This agent orchestrates the ReasoningEngine through a 10-step
    pipeline that mirrors — then extends — the original rule engine.
    """

    def __init__(self) -> None:
        """Instantiate the reasoning engine used across all steps."""
        self.engine = ReasoningEngine()

    def synthesize(
        self,
        task_results: dict,
        memory_context: dict,
        policy: str = "balanced",
        thresholds: dict = None,
    ) -> dict:
        """Produce a complete AgentDecision dict from task outputs.

        Parameters
        ----------
        task_results : dict
            Raw task outputs keyed by task name.
        memory_context : dict
            Recent events from the watchdog / memory store.
        policy : str
            Active policy profile name.

        Returns
        -------
        dict
            Complete decision dict matching the AgentDecision schema,
            with the new ``incident_report``, ``triage_explanation``,
            and ``watchdog_report`` fields.
        """
        try:
            memory_context = memory_context or {}
            task_results = task_results or {}

            # Step 1: build context
            context = self.engine.build_context(task_results)

            # Step 2: extract risk signals (same logic as audio_agent_rules.py)
            signals = self._extract_signals(task_results, thresholds=thresholds)
            risk_signals = self._collect_risk_signals(signals)

            # Step 3: classify event
            event_type, priority, action = self._classify_event(
                signals, risk_signals
            )

            # Step 4: build reasoning chain
            reasoning = self.engine.build_reasoning_chain(
                context, risk_signals, event_type, memory_context
            )

            # Step 5: compute confidence
            successful_tasks = sum(
                1 for r in task_results.values()
                if isinstance(r, dict) and r.get("success", True)
            )
            confidence = self.engine.compute_confidence(
                context, risk_signals, successful_tasks
            )

            # Step 6: build incident report
            incident_report = self.engine.build_incident_report(
                context, event_type, priority, reasoning
            )

            # Step 7: determine recommended_action (refine from classify)
            recommended_action = self._refine_action(
                action, event_type, priority, signals
            )

            # Step 8: privacy mode
            privacy_mode = self._determine_privacy(
                policy, event_type, priority
            )

            # Step 9: data export
            data_export = self._build_data_export(privacy_mode, event_type)

            # Step 10: task health
            task_health = self._task_health(task_results)

            # Build follow-up tasks
            follow_up = self._determine_follow_up(signals, task_results)

            # Triage explanation (populated by orchestrator from TriageAgent)
            triage_explanation = memory_context.get(
                "triage_explanation", "Triage context not available for this run."
            )

            return {
                "event_type": event_type,
                "priority": priority,
                "confidence": confidence,
                "recommended_action": recommended_action,
                "privacy_mode": privacy_mode,
                "models_used": sorted(task_results.keys()),
                "reasoning": reasoning,
                "follow_up_tasks": follow_up,
                "risk_signals": _dedupe(risk_signals),
                "data_export": data_export,
                "task_health": task_health,
                "metadata": {
                    "policy": policy,
                    "speech_ratio": round(signals.get("speech_ratio", 0.0), 4),
                    "quality_label": signals.get("quality_label", "unknown"),
                    "snr_db": round(signals.get("snr_db", 0.0), 2),
                    "transcript": signals.get("transcript", ""),
                },
                "incident_report": incident_report,
                "triage_explanation": triage_explanation,
                "watchdog_report": memory_context.get("watchdog_report", None),
            }

        except Exception as exc:
            # Return a minimal valid decision on any unexpected failure
            return {
                "event_type": "normal_audio",
                "priority": "low",
                "confidence": 0.10,
                "recommended_action": "ignore_or_continue_monitoring",
                "privacy_mode": "metadata_only",
                "models_used": sorted(task_results.keys()) if task_results else [],
                "reasoning": [f"Synthesis failed: {exc}"],
                "follow_up_tasks": [],
                "risk_signals": [],
                "data_export": {"raw_audio": False, "transcript": False,
                                "embeddings": False, "metadata": True},
                "task_health": {},
                "metadata": {"policy": policy if policy else "balanced"},
                "incident_report": "Synthesis pipeline encountered an error.",
                "triage_explanation": "Triage context unavailable due to synthesis error.",
                "watchdog_report": None,
            }

    # ── signal extraction (mirrors audio_agent_rules.py) ─────────────

    def _extract_signals(self, results: dict, thresholds: dict = None) -> dict:
        """Replicate the signal extraction from the original rule engine.

        All thresholds are identical to ``AudioAgent._extract_signals``
        in ``audio_agent_rules.py``, unless dynamic thresholds are passed.
        """
        try:
            vad = results.get("vad", {}) or {}
            quality = results.get("audio_quality_monitoring", {}) or {}
            anomaly = results.get("anomaly_detection", {}) or {}
            emotion = results.get("emotion", {}) or {}
            keyword = results.get("keyword_spotting", {}) or {}
            esc = results.get("esc", {}) or {}
            asr = results.get("asr", {}) or {}
            impulse = results.get("impulse_event", {}) or {}

            speech_ratio = _num(vad.get("speech_ratio", vad.get("percent_speech", 0.0)))
            esc_label = str(esc.get("top_class", "") or "")
            esc_score = _num(esc.get("top_score", 0.0))
            transcript = str(asr.get("text", "") or "").strip()

            speech_ratio_threshold = thresholds.get("speech_ratio_threshold", 0.05) if thresholds else 0.05
            speech_by_vad = bool(
                vad.get("is_speech", vad.get("is_speaking", False))
                or speech_ratio >= speech_ratio_threshold
            )
            speech_by_esc = (
                _looks_like_speech_sound(esc_label)
                and esc_score >= 0.25  # policy.esc_alert_score_threshold
            )
            speech_by_asr = bool(transcript)
            speech_detected = speech_by_vad or speech_by_esc or speech_by_asr

            if speech_by_vad:
                speech_source = "vad"
            elif speech_by_asr:
                speech_source = "asr"
            elif speech_by_esc:
                speech_source = "environmental classifier"
            else:
                speech_source = "none"

            quality_label = str(quality.get("quality_label", "") or "").lower()
            snr_db = _num(quality.get("snr_db", 0.0))
            snr_alert_threshold = thresholds.get("snr_alert_threshold", 8.0) if thresholds else 8.0
            low_quality = quality_label in ("poor",) or (
                quality_label != "" and snr_db < snr_alert_threshold
            )

            emotion_label = str(emotion.get("top_emotion", "") or "").lower()
            emotion_score = _num(emotion.get("top_score", 0.0))
            stress_emotions = {"angry", "fear", "fearful", "sad", "disgust"}
            stress_emotion = (
                emotion_label in stress_emotions
                and emotion_score >= 0.45  # policy.stress_score_threshold
            )

            keyword_label = str(keyword.get("top_label", "") or "")
            keyword_score = _num(keyword.get("top_score", 0.0))
            alert_keywords = {
                "help", "stop", "emergency", "danger", "fire",
                "alarm", "evacuate", "shutdown", "leak", "injured", "pain",
            }
            keyword_is_alert = _contains_alert_keyword(keyword_label, alert_keywords)
            transcript_alert = _contains_alert_keyword(transcript, alert_keywords)
            alert_keyword = bool(
                keyword_is_alert and keyword_score >= 0.50
            )

            esc_event = bool(esc_label and esc_score >= 0.25)

            anomaly_score = _num(anomaly.get("anomaly_score", 0.0))
            anomaly_score_threshold = thresholds.get("anomaly_score_threshold", -0.10) if thresholds else -0.10
            raw_anomaly_flag = bool(
                anomaly.get("is_anomaly", False)
                or anomaly_score < anomaly_score_threshold
            )
            benign_speech_context = (
                raw_anomaly_flag
                and speech_detected
                and _looks_like_speech_sound(esc_label)
                and not _looks_like_alert_sound(esc_label)
                and not low_quality
            )
            anomaly_flag = raw_anomaly_flag and not benign_speech_context

            impulse_flag = bool(impulse.get("is_impulse", False))
            impulse_label = str(impulse.get("event_label", "") or "")
            impulse_confidence = _num(impulse.get("confidence", 0.0))

            return {
                "speech_detected": speech_detected,
                "speech_source": speech_source,
                "speech_ratio": speech_ratio,
                "low_quality": low_quality,
                "quality_label": quality_label or "unknown",
                "snr_db": round(snr_db, 2),
                "anomaly": anomaly_flag,
                "raw_anomaly": raw_anomaly_flag,
                "anomaly_downgraded": benign_speech_context,
                "anomaly_score": anomaly_score,
                "impulse_event": impulse_flag,
                "impulse_label": impulse_label,
                "impulse_confidence": impulse_confidence,
                "stress_emotion": stress_emotion,
                "emotion_label": emotion_label,
                "emotion_score": emotion_score,
                "alert_keyword": alert_keyword,
                "transcript_alert": transcript_alert,
                "keyword_label": keyword_label,
                "keyword_score": keyword_score,
                "esc_event": esc_event,
                "esc_label": esc_label,
                "esc_score": esc_score,
                "transcript": transcript,
                "is_high_risk": (
                    anomaly_flag
                    or impulse_flag
                    or stress_emotion
                    or alert_keyword
                    or transcript_alert
                    or _looks_like_alert_sound(esc_label)
                ),
            }
        except Exception:
            return {
                "speech_detected": False, "speech_source": "none",
                "speech_ratio": 0.0, "low_quality": False,
                "quality_label": "unknown", "snr_db": 0.0,
                "anomaly": False, "raw_anomaly": False,
                "anomaly_downgraded": False, "anomaly_score": 0.0,
                "impulse_event": False, "impulse_label": "",
                "impulse_confidence": 0.0, "stress_emotion": False,
                "emotion_label": "", "emotion_score": 0.0,
                "alert_keyword": False, "transcript_alert": False,
                "keyword_label": "", "keyword_score": 0.0,
                "esc_event": False, "esc_label": "", "esc_score": 0.0,
                "transcript": "", "is_high_risk": False,
            }

    def _collect_risk_signals(self, signals: dict) -> List[str]:
        """Build a list of risk signal identifiers from extracted signals."""
        try:
            risk: List[str] = []
            if signals.get("low_quality"):
                risk.append("low_audio_quality")
            if signals.get("anomaly"):
                risk.append("audio_anomaly")
            if signals.get("impulse_event"):
                risk.append("impulse_event")
            if signals.get("stress_emotion"):
                risk.append("stressed_speech")
            if signals.get("alert_keyword"):
                risk.append("alert_keyword")
            if signals.get("transcript_alert"):
                risk.append("transcript_alert")
            if signals.get("esc_event") and _looks_like_alert_sound(signals.get("esc_label", "")):
                risk.append("alert_sound")
            return risk
        except Exception:
            return []

    # ── event classification (matches audio_agent_rules.py) ──────────

    def _classify_event(
        self,
        signals: dict,
        risk_signals: List[str],
    ) -> Tuple[str, str, str]:
        """Classify event using the spec's priority order."""
        try:
            risk_count = len(set(risk_signals))

            if signals.get("impulse_event"):
                return "possible_impulse_threat", "high", "create_incident_and_notify_operator"
            if risk_count >= 2:
                return "possible_safety_incident", "high", "create_incident_and_notify_operator"
            if "alert_sound" in risk_signals or "audio_anomaly" in risk_signals:
                return "acoustic_risk_event", "medium", "store_metadata_and_request_review"
            if signals.get("transcript"):
                return "speech_workflow_event", "medium", "extract_intent_and_route_to_workflow"
            if signals.get("speech_detected"):
                return "speech_detected", "low", "log_event"
            if signals.get("esc_event"):
                return "environmental_audio_event", "low", "log_event"
            return "normal_audio", "low", "ignore_or_continue_monitoring"
        except Exception:
            return "normal_audio", "low", "ignore_or_continue_monitoring"

    def _refine_action(
        self,
        base_action: str,
        event_type: str,
        priority: str,
        signals: dict,
    ) -> str:
        """Refine the recommended action based on full signal context."""
        try:
            if priority == "high":
                return "create_incident_and_notify_operator"
            if priority == "medium" and event_type == "speech_workflow_event":
                return "extract_intent_and_route_to_workflow"
            if priority == "medium":
                return "store_metadata_and_request_review"
            if signals.get("speech_detected"):
                return "log_event"
            return "ignore_or_continue_monitoring"
        except Exception:
            return base_action

    # ── privacy ──────────────────────────────────────────────────────

    def _determine_privacy(
        self,
        policy: str,
        event_type: str,
        priority: str,
    ) -> str:
        """Determine privacy mode based on policy and event priority."""
        try:
            if policy == "privacy_first":
                return "metadata_only"
            if priority == "high":
                return "metadata_plus_redacted_evidence"
            return "metadata_only"
        except Exception:
            return "metadata_only"

    def _build_data_export(
        self,
        privacy_mode: str,
        event_type: str,
    ) -> dict:
        """Build the data export dict based on privacy mode."""
        try:
            if privacy_mode == "metadata_plus_redacted_evidence":
                return {
                    "raw_audio": False,
                    "transcript": True,
                    "embeddings": False,
                    "metadata": True,
                }
            return {
                "raw_audio": False,
                "transcript": event_type == "speech_workflow_event",
                "embeddings": False,
                "metadata": True,
            }
        except Exception:
            return {
                "raw_audio": False, "transcript": False,
                "embeddings": False, "metadata": True,
            }

    # ── task health ──────────────────────────────────────────────────

    def _task_health(self, results: dict) -> dict:
        """Compute per-task health status."""
        try:
            health = {}
            for name, result in (results or {}).items():
                if isinstance(result, dict) and result.get("success", True):
                    health[name] = "ok"
                else:
                    err = ""
                    if isinstance(result, dict):
                        err = str(result.get("error", "failed"))[:120]
                    health[name] = err or "failed"
            return health
        except Exception:
            return {}

    # ── follow-up tasks ──────────────────────────────────────────────

    def _determine_follow_up(
        self,
        signals: dict,
        already_run: dict,
    ) -> List[str]:
        """Determine which follow-up tasks to recommend."""
        try:
            already = set(already_run.keys()) if already_run else set()
            follow: List[str] = []

            if signals.get("speech_detected"):
                for t in ("asr", "emotion"):
                    if t not in already:
                        follow.append(t)
            else:
                if "esc" not in already:
                    follow.append("esc")

            if signals.get("is_high_risk"):
                for t in ("esc", "keyword_spotting", "speaker_id"):
                    if t not in already:
                        follow.append(t)

            if signals.get("low_quality") and "speech_quality" not in already:
                follow.append("speech_quality")

            return _dedupe(follow)
        except Exception:
            return []


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


def _num(value) -> float:
    """Safely coerce *value* to ``float``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _looks_like_alert_sound(label: str) -> bool:
    """Check if an ESC label looks like an alert / emergency sound."""
    label_l = (label or "").lower()
    alert_terms = [
        "alarm", "siren", "explosion", "gunshot", "glass",
        "breaking", "crash", "bang", "scream", "shout",
        "emergency", "smoke detector", "fire",
    ]
    return any(term in label_l for term in alert_terms)


def _looks_like_speech_sound(label: str) -> bool:
    """Check if an ESC label looks like human speech."""
    label_l = (label or "").lower()
    speech_terms = [
        "speech", "conversation", "narration", "speech synthesizer",
        "male speech", "female speech", "child speech", "talking",
    ]
    return any(term in label_l for term in speech_terms)


def _contains_alert_keyword(text: str, alert_keywords) -> bool:
    """Check if *text* contains any alert keyword (word-level match)."""
    text_l = f" {text or ''} ".lower()
    normalised = "".join(ch if ch.isalnum() else " " for ch in text_l)
    words = set(normalised.split())
    return any(kw.lower() in words for kw in alert_keywords)
