"""Deterministic chain-of-thought reasoning engine for edge audio intelligence.

This module produces structured natural-language reasoning from raw task
outputs.  It does NOT use any LLM, API call, or external model — every
computation is pure Python with sub-millisecond latency.

Classes
-------
ReasoningEngine
    Shared utility consumed by TriageAgent, SynthesisAgent, and WatchdogAgent.
"""

from __future__ import annotations

from typing import Any, Dict, List


class ReasoningEngine:
    """Produces structured chain-of-thought reasoning from task outputs.

    All methods are deterministic and stateless — safe for concurrent use
    across threads without locks.
    """

    # ── context building ─────────────────────────────────────────────

    def build_context(self, task_results: dict) -> dict:
        """Extract and normalise every signal into a flat context dict.

        Parameters
        ----------
        task_results : dict
            Raw task outputs keyed by task name.  Missing keys are handled
            gracefully so partial runs still produce valid context.

        Returns
        -------
        dict
            Flat dict with canonical signal keys for downstream reasoning.
        """
        try:
            vad = task_results.get("vad", {}) or {}
            asr = task_results.get("asr", {}) or {}
            emotion = task_results.get("emotion", {}) or {}
            esc = task_results.get("esc", {}) or {}
            anomaly = task_results.get("anomaly_detection", {}) or {}
            impulse = task_results.get("impulse_event", {}) or {}
            quality = task_results.get("audio_quality_monitoring", {}) or {}
            speaker = task_results.get("speaker_id", {}) or {}
            lang = task_results.get("lang_accent_id", {}) or {}
            keyword = task_results.get("keyword_spotting", {}) or {}
            music = task_results.get("music_speech_detection", {}) or {}
            genre = task_results.get("music_genre", {}) or {}

            speech_ratio = _num(vad.get("speech_ratio", 0.0))
            is_speech = bool(
                vad.get("is_speech", False)
                or vad.get("is_speaking", False)
                or speech_ratio >= 0.05
            )
            speech_duration_sec = _num(vad.get("total_speech_sec", vad.get("speech_duration_sec", 0.0)))

            transcript = str(asr.get("text", "") or "").strip()
            speech_detected = bool(
                is_speech
                or bool(transcript)
            )

            transcript = str(asr.get("text", "") or "").strip()
            keywords_raw = keyword.get("top_label", "")
            esc_class = str(esc.get("top_class", "") or "")
            esc_score = _num(esc.get("top_score", 0.0))
            emotion_label = str(emotion.get("top_emotion", "") or "").lower()
            emotion_score = _num(emotion.get("top_score", 0.0))
            emotion_skipped = bool(emotion.get("skipped", False))
            emotion_low_confidence = bool(emotion.get("low_confidence", False))
            emotion_low_confidence_note = str(emotion.get("low_confidence_note", "") or "")
            anomaly_flag = bool(anomaly.get("is_anomaly", False))
            anomaly_score = _num(anomaly.get("anomaly_score", 0.0))
            impulse_flag = bool(impulse.get("is_impulse", False))
            impulse_label = str(impulse.get("event_label", "") or "")
            quality_label = str(quality.get("quality_label", "") or "").lower()
            snr_db = _num(quality.get("snr_db", 0.0))
            num_speakers = int(_num(speaker.get("num_speakers", 0)))
            language = str(lang.get("top_language", "") or "")
            genre_label = str(genre.get("top_genre", "") or "")
            music_fraction = _num(music.get("music_fraction", 0.0))
            music_detected = music_fraction > 0.5

            return {
                "speech_detected": speech_detected,
                "speech_ratio": round(speech_ratio, 4),
                "is_speech": is_speech,
                "speech_duration_sec": round(speech_duration_sec, 3),
                "emotion": emotion_label,
                "emotion_score": round(emotion_score, 4),
                "emotion_skipped": emotion_skipped,
                "emotion_low_confidence": emotion_low_confidence,
                "emotion_low_confidence_note": emotion_low_confidence_note,
                "transcript": transcript,
                "keywords": str(keywords_raw or ""),
                "esc_class": esc_class,
                "esc_score": round(esc_score, 4),
                "anomaly": anomaly_flag,
                "anomaly_score": round(anomaly_score, 4),
                "impulse": impulse_flag,
                "impulse_label": impulse_label,
                "quality_label": quality_label or "unknown",
                "snr_db": round(snr_db, 2),
                "num_speakers": num_speakers,
                "language": language,
                "genre": genre_label,
                "music_detected": music_detected,
            }
        except Exception:
            # Absolute fallback — return a safe empty context
            return {
                "speech_detected": False,
                "speech_ratio": 0.0,
                "is_speech": False,
                "speech_duration_sec": 0.0,
                "emotion": "",
                "emotion_score": 0.0,
                "emotion_skipped": False,
                "emotion_low_confidence": False,
                "emotion_low_confidence_note": "",
                "transcript": "",
                "keywords": "",
                "esc_class": "",
                "esc_score": 0.0,
                "anomaly": False,
                "anomaly_score": 0.0,
                "impulse": False,
                "impulse_label": "",
                "quality_label": "unknown",
                "snr_db": 0.0,
                "num_speakers": 0,
                "language": "",
                "genre": "",
                "music_detected": False,
            }

    # ── chain-of-thought reasoning ───────────────────────────────────

    def build_reasoning_chain(
        self,
        context: dict,
        risk_signals: list,
        event_type: str,
        memory_context: dict,
    ) -> List[str]:
        """Produce a list of human-readable reasoning steps.

        Each string reads like a sentence an analyst would write, not a
        debug log line.  The chain covers every active signal and ends
        with a confidence justification.

        Parameters
        ----------
        context : dict
            Flat context dict from ``build_context``.
        risk_signals : list
            Active risk signal identifiers.
        event_type : str
            Classified event type.
        memory_context : dict
            Recent memory events from the watchdog / memory store.

        Returns
        -------
        list[str]
            Ordered chain-of-thought reasoning steps.
        """
        try:
            chain: List[str] = []

            # ── Speech / VAD ───────────────────────────────────────
            ratio_pct = round(context.get("speech_ratio", 0.0) * 100, 1)
            speech_ratio = context.get("speech_ratio", 0.0)
            if speech_ratio > 0.3:
                chain.append(
                    f"VAD detected speech at {ratio_pct}% ratio "
                    f"— activating speech analysis pipeline"
                )
            elif speech_ratio > 0.0:
                chain.append(
                    f"VAD detected low speech activity at {ratio_pct}%"
                )
            else:
                chain.append(
                    "VAD detected no speech — routing to environmental analysis pipeline"
                )

            # ── Emotion ────────────────────────────────────────────
            if context.get("emotion_skipped"):
                chain.append(
                    "Emotion analysis skipped — insufficient "
                    "speech detected for reliable classification"
                )
            elif context.get("emotion_low_confidence"):
                emo = context.get("emotion", "")
                emo_score = context.get("emotion_score", 0.0)
                pct = round(emo_score * 100, 1)
                note = context.get("emotion_low_confidence_note", "")
                chain.append(
                    f"Emotion classifier returned "
                    f"{emo} at {pct}% — below "
                    f"confidence threshold, treating as neutral. "
                    f"Note: {note}"
                )
            else:
                emo = context.get("emotion", "")
                emo_score = context.get("emotion_score", 0.0)
                if emo and emo_score > 0:
                    pct = round(emo_score * 100, 1)
                    if emo in {"angry", "fear", "fearful", "sad", "disgust"}:
                        chain.append(
                            f"Emotion classifier returned {emo} at {pct}% confidence "
                            f"— stress signal raised"
                        )
                    else:
                        chain.append(
                            f"Emotion classifier returned {emo} at {pct}% confidence "
                            f"— no stress indicated"
                        )

            # ── Transcript / ASR ───────────────────────────────────
            transcript = context.get("transcript", "")
            if transcript:
                preview = transcript[:80] + ("…" if len(transcript) > 80 else "")
                chain.append(
                    f"ASR produced transcript: \"{preview}\""
                )
                # Check for alert keywords
                alert_words = {
                    "help", "stop", "emergency", "danger", "fire",
                    "alarm", "evacuate", "shutdown", "leak", "injured", "pain",
                }
                found = [w for w in alert_words if w in transcript.lower().split()]
                if found:
                    chain.append(
                        f"ASR transcript contains keyword '{found[0]}' "
                        f"— transcript_alert confirmed"
                    )

            # ── Keywords ───────────────────────────────────────────
            kw = context.get("keywords", "")
            if kw:
                chain.append(
                    f"Keyword spotter detected '{kw}' in the audio stream"
                )

            # ── ESC ────────────────────────────────────────────────
            esc = context.get("esc_class", "")
            esc_score = context.get("esc_score", 0.0)
            if esc and esc_score > 0:
                pct = round(esc_score * 100, 1)
                chain.append(
                    f"Environmental sound classifier identified '{esc}' "
                    f"at {pct}% confidence"
                )

            # ── Anomaly ────────────────────────────────────────────
            if context.get("anomaly"):
                score = round(context.get("anomaly_score", 0.0), 4)
                chain.append(
                    f"Anomaly detector flagged unusual audio characteristics "
                    f"(score: {score})"
                )

            # ── Impulse ────────────────────────────────────────────
            if context.get("impulse"):
                label = context.get("impulse_label", "unknown")
                chain.append(
                    f"Impulse event detector triggered — classified as '{label}'"
                )

            # ── Risk signal aggregation ────────────────────────────
            n_risk = len(risk_signals)
            if n_risk >= 2:
                chain.append(
                    f"{n_risk} concurrent risk signals detected "
                    f"({', '.join(risk_signals)}) — escalating to {event_type}"
                )
            elif n_risk == 1:
                chain.append(
                    f"Single risk signal '{risk_signals[0]}' active "
                    f"— classifying as {event_type}"
                )

            # ── Memory context reasoning ───────────────────────────
            recent_events = memory_context.get("recent_events", [])
            if not recent_events:
                recent_events = memory_context.get("related_events", [])
            if recent_events:
                last = recent_events[-1] if recent_events else {}
                last_type = last.get("event_type", "unknown")
                last_ts = last.get("timestamp", "")
                chain.append(
                    f"Memory shows '{last_type}' event in recent history "
                    f"— timeline context incorporated"
                )
                escalation = memory_context.get("escalation_signals", [])
                if escalation:
                    chain.append(
                        f"Memory escalation cues: {', '.join(escalation)} "
                        f"— timeline_escalation added"
                    )

            # ── Audio quality reasoning ────────────────────────────
            ql = context.get("quality_label", "")
            snr = context.get("snr_db", 0.0)
            if ql in ("poor", "fair"):
                chain.append(
                    f"Audio quality is '{ql}' with SNR {snr:.1f} dB "
                    f"— confidence penalty applied"
                )
            elif snr < 8 and snr > 0:
                chain.append(
                    f"Low SNR ({snr:.1f} dB) may degrade model accuracy "
                    f"— confidence adjusted downward"
                )

            # ── Multi-speaker reasoning ────────────────────────────
            ns = context.get("num_speakers", 0)
            if ns > 1:
                chain.append(
                    f"{ns} distinct speakers detected "
                    f"— multi-party interaction noted"
                )

            # ── Music reasoning ────────────────────────────────────
            if context.get("music_detected"):
                genre = context.get("genre", "")
                suffix = f" (genre: {genre})" if genre else ""
                chain.append(
                    f"Music detected in the audio stream{suffix} "
                    f"— ambient audio classification applied"
                )

            # ── Multi-source confirmation ──────────────────────────
            confirm_sources = []
            if context.get("emotion") and context.get("emotion_score", 0) > 0.5:
                confirm_sources.append("emotion")
            if context.get("keywords"):
                confirm_sources.append("keyword")
            if context.get("transcript"):
                confirm_sources.append("transcript")
            if context.get("esc_class"):
                confirm_sources.append("ESC")
            if len(confirm_sources) >= 3:
                chain.append(
                    f"Multi-source confirmation ({' + '.join(confirm_sources)}) "
                    f"— confidence boosted"
                )

            # ── Confidence justification (always last) ─────────────
            chain.append(
                f"Final classification: {event_type} — "
                f"confidence computed from {len(confirm_sources)} corroborating "
                f"signals and {n_risk} risk indicator(s)"
            )

            return chain if chain else ["No actionable signals detected in this window"]

        except Exception:
            return ["Reasoning chain could not be constructed — using default classification"]

    # ── confidence scoring ───────────────────────────────────────────

    def compute_confidence(
        self,
        context: dict,
        risk_signals: list,
        successful_tasks: int,
    ) -> float:
        """Compute a deterministic confidence score from context signals.

        Parameters
        ----------
        context : dict
            Flat context dict from ``build_context``.
        risk_signals : list
            Active risk signal identifiers.
        successful_tasks : int
            Number of tasks that completed without error.

        Returns
        -------
        float
            Confidence score clamped to [0.10, 0.98].
        """
        try:
            score = 0.35

            # +0.08 per successful task, capped at 0.25
            task_contrib = min(successful_tasks * 0.08, 0.25)
            score += task_contrib

            # +0.10 if speech detected
            if context.get("speech_detected"):
                score += 0.10

            # +0.08 if transcript non-empty
            if context.get("transcript"):
                score += 0.08

            # +0.06 if ESC class detected
            if context.get("esc_class"):
                score += 0.06

            # +0.05 per risk signal, capped at 0.20
            risk_contrib = min(len(risk_signals) * 0.05, 0.20)
            score += risk_contrib

            # +0.07 if emotion score > 0.7
            if context.get("emotion_score", 0.0) > 0.7:
                score += 0.07

            # -0.10 if quality_label is poor
            if context.get("quality_label", "").lower() == "poor":
                score -= 0.10

            # -0.05 if snr_db < 8
            if context.get("snr_db", 99.0) < 8:
                score -= 0.05

            return round(max(0.10, min(score, 0.98)), 2)

        except Exception:
            return 0.35

    # ── incident report ──────────────────────────────────────────────

    def build_incident_report(
        self,
        context: dict,
        event_type: str,
        priority: str,
        reasoning: List[str],
    ) -> str:
        """Generate an operator-readable incident report paragraph.

        Parameters
        ----------
        context : dict
            Flat context dict from ``build_context``.
        event_type : str
            Classified event type.
        priority : str
            Priority level (low / medium / high).
        reasoning : list[str]
            Chain-of-thought reasoning steps.

        Returns
        -------
        str
            Natural language paragraph (2-4 sentences) suitable for an
            operator dashboard or alert notification.
        """
        try:
            parts: List[str] = []

            # Opening sentence — what happened
            event_labels = {
                "possible_safety_incident": "a possible safety incident",
                "possible_impulse_threat": "a possible impulse threat",
                "acoustic_risk_event": "an acoustic risk event",
                "speech_workflow_event": "a speech workflow event",
                "speech_detected": "speech activity",
                "environmental_audio_event": "an environmental audio event",
                "normal_audio": "normal ambient audio",
            }
            what = event_labels.get(event_type, event_type.replace("_", " "))
            parts.append(
                f"Audio analysis detected {what} in the monitoring zone."
            )

            # Detail sentence — which signals fired
            details: List[str] = []
            if context.get("speech_detected"):
                emo = context.get("emotion", "")
                emo_pct = round(context.get("emotion_score", 0.0) * 100, 1)
                if emo and emo_pct > 0:
                    details.append(
                        f"Worker speech was detected with {emo} emotion "
                        f"at {emo_pct}% confidence"
                    )
                else:
                    ratio_pct = round(context.get("speech_ratio", 0.0) * 100, 1)
                    details.append(f"Speech was detected at {ratio_pct}% ratio")

            transcript = context.get("transcript", "")
            if transcript:
                alert_words = {
                    "help", "stop", "emergency", "danger", "fire",
                    "alarm", "evacuate", "shutdown", "leak", "injured", "pain",
                }
                found = [w for w in alert_words if w in transcript.lower().split()]
                if found:
                    details.append(
                        f"the keyword '{found[0]}' was confirmed in the transcript"
                    )

            esc = context.get("esc_class", "")
            esc_score = context.get("esc_score", 0.0)
            if esc:
                pct = round(esc_score * 100, 1)
                details.append(
                    f"Environmental sound classifier flagged '{esc}' at {pct}% confidence"
                )

            if context.get("impulse"):
                label = context.get("impulse_label", "impact")
                details.append(f"an impulse event ('{label}') was detected")

            if context.get("anomaly"):
                details.append("anomaly detector flagged unusual audio characteristics")

            if details:
                parts.append(" and ".join(details[:3]).capitalize() + ".")

            # Closing sentence — recommendation
            if priority == "high":
                parts.append("Immediate operator review is recommended.")
            elif priority == "medium":
                parts.append("Operator review at next available opportunity is suggested.")
            else:
                parts.append("No immediate action is required; event has been logged.")

            return " ".join(parts)

        except Exception:
            return (
                f"Audio analysis detected an event classified as {event_type} "
                f"with {priority} priority. Review the reasoning chain for details."
            )


# ── private helpers ──────────────────────────────────────────────────

def _num(value) -> float:
    """Safely coerce *value* to ``float``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
