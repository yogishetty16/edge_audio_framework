"""Agentic edge audio orchestrator.

This layer treats pretrained task models as tools. It does not retrain or modify
them; it plans which tools should run and converts their outputs into a single
enterprise-grade decision.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .policy import get_policy
from .schema import AgentDecision, AgentPolicy, TaskPlan


class AudioAgent:
    """Rule-governed agent for edge-first speech and audio intelligence."""

    def __init__(self, profile: str = "balanced", policy: Optional[AgentPolicy] = None) -> None:
        self.policy = policy or get_policy(profile)

    def plan_initial_tasks(self, requested_tasks: Optional[Iterable[str]] = None) -> TaskPlan:
        """Return the first lightweight task batch."""
        allowed = _clean_task_list(requested_tasks)
        tasks = list(self.policy.always_on_tasks)
        skipped = {}

        if allowed:
            skipped = {t: "not requested by caller" for t in tasks if t not in allowed}
            tasks = [t for t in tasks if t in allowed]

        if not tasks and allowed:
            tasks = list(allowed[: self.policy.max_parallel_tasks])

        return TaskPlan(
            tasks=_dedupe(tasks)[: self.policy.max_parallel_tasks],
            reason="Initial edge triage: run lightweight signal, quality, and risk checks first.",
            skipped_tasks=skipped,
        )

    def plan_follow_up_tasks(
        self,
        results: Dict[str, Dict[str, Any]],
        requested_tasks: Optional[Iterable[str]] = None,
        already_run: Optional[Iterable[str]] = None,
    ) -> TaskPlan:
        """Plan the next task batch from first-pass results."""
        allowed = _clean_task_list(requested_tasks)
        already = set(_clean_task_list(already_run))
        signals = self._extract_signals(results)
        followups: List[str] = []
        reasons: List[str] = []

        if signals["speech_detected"]:
            followups.extend(self.policy.speech_follow_up_tasks)
            reasons.append("speech was detected")
        else:
            followups.extend(self.policy.non_speech_follow_up_tasks)
            reasons.append("speech was not dominant")

        if signals["is_high_risk"]:
            followups.extend(self.policy.high_risk_follow_up_tasks)
            reasons.append("risk signals crossed policy thresholds")

        if signals["low_quality"]:
            followups.append("speech_quality")
            reasons.append("audio quality is weak")

        tasks = [t for t in _dedupe(followups) if t not in already]
        skipped: Dict[str, str] = {}
        if allowed:
            skipped = {t: "not requested by caller" for t in tasks if t not in allowed}
            tasks = [t for t in tasks if t in allowed]

        return TaskPlan(
            tasks=tasks[: self.policy.max_parallel_tasks],
            reason="Follow-up based on " + ", ".join(reasons) + ".",
            skipped_tasks=skipped,
        )

    def decide(self, results: Dict[str, Dict[str, Any]]) -> AgentDecision:
        """Create one actionable decision from task outputs."""
        normalized = {name: _as_dict(value) for name, value in (results or {}).items()}
        signals = self._extract_signals(normalized)
        task_health = _task_health(normalized)
        reasoning: List[str] = []
        risk_signals: List[str] = []

        if signals["speech_detected"]:
            if signals["speech_source"] == "vad":
                reasoning.append(f"speech detected at {signals['speech_ratio'] * 100:.1f}% of the window")
            else:
                reasoning.append(
                    f"speech indicated by {signals['speech_source']} despite low VAD ratio "
                    f"({signals['speech_ratio'] * 100:.1f}%)"
                )
        else:
            reasoning.append("speech was not dominant in the window")

        if signals["low_quality"]:
            reasoning.append("audio quality is below policy target")
            risk_signals.append("low_audio_quality")

        if signals["anomaly"]:
            reasoning.append("anomaly detector flagged unusual audio characteristics")
            risk_signals.append("audio_anomaly")
        elif signals["anomaly_downgraded"]:
            reasoning.append("isolated anomaly signal was downgraded because speech/quality evidence was benign")

        if signals["impulse_event"]:
            reasoning.append(
                f"short impulse event detected: {signals['impulse_label']} "
                f"({signals['impulse_confidence'] * 100:.1f}%)"
            )
            risk_signals.append("impulse_event")

        if signals["stress_emotion"]:
            reasoning.append(
                f"stress emotion detected: {signals['emotion_label']} "
                f"({signals['emotion_score'] * 100:.1f}%)"
            )
            risk_signals.append("stressed_speech")

        if signals["alert_keyword"]:
            reasoning.append(
                f"keyword spotted: {signals['keyword_label']} "
                f"({signals['keyword_score'] * 100:.1f}%)"
            )
            risk_signals.append("alert_keyword")
        elif signals["keyword_label"]:
            reasoning.append(
                f"non-alert keyword spotted: {signals['keyword_label']} "
                f"({signals['keyword_score'] * 100:.1f}%)"
            )

        if signals["transcript_alert"]:
            reasoning.append("transcript contains explicit safety or distress language")
            risk_signals.append("transcript_alert")

        if signals["esc_event"]:
            reasoning.append(
                f"environmental sound detected: {signals['esc_label']} "
                f"({signals['esc_score'] * 100:.1f}%)"
            )
            if _looks_like_alert_sound(signals["esc_label"]):
                risk_signals.append("alert_sound")

        if signals["transcript"]:
            reasoning.append("transcript is available for structured workflow extraction")

        event_type, priority, action = self._classify_event(signals, risk_signals)
        confidence = self._estimate_confidence(signals, task_health, risk_signals)
        privacy_mode, data_export = self._privacy_decision(event_type, priority)

        follow_up = self.plan_follow_up_tasks(normalized, already_run=normalized.keys()).tasks

        return AgentDecision(
            event_type=event_type,
            priority=priority,
            confidence=confidence,
            recommended_action=action,
            privacy_mode=privacy_mode,
            models_used=sorted(normalized.keys()),
            reasoning=reasoning or ["no reliable signal available"],
            follow_up_tasks=follow_up,
            risk_signals=_dedupe(risk_signals),
            data_export=data_export,
            task_health=task_health,
            metadata={
                "policy": self.policy.name,
                "speech_ratio": round(signals["speech_ratio"], 4),
                "quality_label": signals["quality_label"],
                "snr_db": signals["snr_db"],
                "transcript": signals["transcript"],
            },
        )

    def _extract_signals(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        vad = results.get("vad", {})
        quality = results.get("audio_quality_monitoring", {})
        anomaly = results.get("anomaly_detection", {})
        emotion = results.get("emotion", {})
        keyword = results.get("keyword_spotting", {})
        esc = results.get("esc", {})
        asr = results.get("asr", {})
        impulse = results.get("impulse_event", {})

        speech_ratio = _number(vad.get("speech_ratio", vad.get("percent_speech", 0.0)))
        esc_label = str(esc.get("top_class", "") or "")
        esc_score = _number(esc.get("top_score", 0.0))
        transcript = str(asr.get("text", "") or "").strip()
        speech_by_vad = bool(
            vad.get("is_speech", vad.get("is_speaking", False))
            or speech_ratio >= self.policy.min_speech_ratio
        )
        speech_by_esc = _looks_like_speech_sound(esc_label) and esc_score >= self.policy.esc_alert_score_threshold
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
        snr_db = _number(quality.get("snr_db", 0.0))
        low_quality = quality_label in self.policy.low_quality_labels or (
            quality_label != "" and snr_db < self.policy.low_quality_snr_db
        )

        emotion_label = str(emotion.get("top_emotion", "") or "").lower()
        emotion_score = _number(emotion.get("top_score", 0.0))
        stress_emotion = (
            emotion_label in self.policy.stress_emotions
            and emotion_score >= self.policy.stress_score_threshold
        )

        keyword_label = str(keyword.get("top_label", "") or "")
        keyword_score = _number(keyword.get("top_score", 0.0))
        keyword_is_alert = _contains_alert_keyword(keyword_label, self.policy.alert_keywords)
        transcript_alert = _contains_alert_keyword(transcript, self.policy.alert_keywords)
        alert_keyword = bool(
            keyword_is_alert
            and keyword_score >= self.policy.keyword_alert_score_threshold
        )

        esc_event = bool(esc_label and esc_score >= self.policy.esc_alert_score_threshold)

        anomaly_score = _number(anomaly.get("anomaly_score", 0.0))
        raw_anomaly_flag = bool(
            anomaly.get("is_anomaly", False)
            or anomaly_score < self.policy.anomaly_score_threshold
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
        impulse_confidence = _number(impulse.get("confidence", 0.0))

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

    def _classify_event(self, signals: Dict[str, Any], risk_signals: List[str]) -> Tuple[str, str, str]:
        risk_count = len(set(risk_signals))

        if risk_count >= 2:
            return "possible_safety_incident", "high", "create_incident_and_notify_operator"
        if "impulse_event" in risk_signals:
            return "possible_impulse_threat", "high", "create_incident_and_notify_operator"
        if "audio_anomaly" in risk_signals or "alert_sound" in risk_signals:
            return "acoustic_risk_event", "medium", "store_metadata_and_request_review"
        if signals["speech_detected"] and signals["transcript"]:
            return "speech_workflow_event", "medium", "extract_intent_and_route_to_workflow"
        if signals["speech_detected"]:
            return "speech_detected", "low", "wait_for_more_context"
        if signals["esc_event"]:
            return "environmental_audio_event", "low", "log_event"
        return "normal_audio", "low", "ignore_or_continue_monitoring"

    def _estimate_confidence(
        self,
        signals: Dict[str, Any],
        task_health: Dict[str, str],
        risk_signals: List[str],
    ) -> float:
        score = 0.35
        successful_tasks = sum(1 for status in task_health.values() if status == "ok")
        score += min(successful_tasks, 5) * 0.08

        if signals["speech_detected"]:
            score += min(signals["speech_ratio"], 1.0) * 0.12
        if signals["transcript"]:
            score += 0.10
        if signals["esc_event"]:
            score += min(signals["esc_score"], 1.0) * 0.08
        if signals["stress_emotion"]:
            score += min(signals["emotion_score"], 1.0) * 0.08
        if risk_signals:
            score += min(len(set(risk_signals)), 3) * 0.05
        if signals["low_quality"]:
            score -= 0.12

        return round(max(0.0, min(score, 0.98)), 2)

    def _privacy_decision(self, event_type: str, priority: str) -> Tuple[str, Dict[str, bool]]:
        if self.policy.allow_raw_audio_export and not self.policy.strict_privacy:
            return "raw_audio_allowed", {
                "raw_audio": True,
                "transcript": True,
                "embeddings": True,
                "metadata": True,
            }

        if priority == "high":
            return "metadata_plus_redacted_evidence", {
                "raw_audio": False,
                "transcript": True,
                "embeddings": False,
                "metadata": True,
            }

        return "metadata_only", {
            "raw_audio": False,
            "transcript": event_type == "speech_workflow_event",
            "embeddings": False,
            "metadata": True,
        }


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def _clean_task_list(tasks: Optional[Iterable[str]]) -> List[str]:
    if not tasks:
        return []
    return _dedupe([str(t).strip() for t in tasks if str(t).strip()])


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _task_health(results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    health = {}
    for name, result in results.items():
        if result.get("success", True):
            health[name] = "ok"
        else:
            health[name] = str(result.get("error", "failed"))[:120]
    return health


def _looks_like_alert_sound(label: str) -> bool:
    label_l = (label or "").lower()
    alert_terms = [
        "alarm",
        "siren",
        "explosion",
        "gunshot",
        "glass",
        "breaking",
        "crash",
        "bang",
        "scream",
        "shout",
        "emergency",
        "smoke detector",
        "fire",
    ]
    return any(term in label_l for term in alert_terms)


def _looks_like_speech_sound(label: str) -> bool:
    label_l = (label or "").lower()
    speech_terms = [
        "speech",
        "conversation",
        "narration",
        "speech synthesizer",
        "male speech",
        "female speech",
        "child speech",
        "talking",
    ]
    return any(term in label_l for term in speech_terms)


def _contains_alert_keyword(text: str, alert_keywords: Iterable[str]) -> bool:
    text_l = f" {text or ''} ".lower()
    normalized = "".join(ch if ch.isalnum() else " " for ch in text_l)
    words = set(normalized.split())
    return any(keyword.lower() in words for keyword in alert_keywords)
