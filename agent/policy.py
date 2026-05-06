"""Policy presets for enterprise edge audio deployments."""

from __future__ import annotations

from .schema import AgentPolicy


PROFILE_POLICIES = {
    "balanced": AgentPolicy(
        name="balanced",
        always_on_tasks=["vad", "audio_quality_monitoring", "anomaly_detection", "impulse_event"],
        speech_follow_up_tasks=["asr", "emotion"],
        non_speech_follow_up_tasks=["esc"],
        high_risk_follow_up_tasks=["esc", "keyword_spotting", "speaker_id"],
        strict_privacy=True,
        max_parallel_tasks=6,
    ),
    "industrial_safety": AgentPolicy(
        name="industrial_safety",
        always_on_tasks=["vad", "audio_quality_monitoring", "anomaly_detection", "impulse_event", "esc"],
        speech_follow_up_tasks=["asr", "emotion", "keyword_spotting"],
        non_speech_follow_up_tasks=["esc", "acoustic_event_detection"],
        high_risk_follow_up_tasks=["speaker_id", "speech_quality"],
        strict_privacy=True,
        max_parallel_tasks=6,
        esc_alert_score_threshold=0.20,
        stress_score_threshold=0.40,
    ),
    "privacy_first": AgentPolicy(
        name="privacy_first",
        always_on_tasks=["vad", "audio_quality_monitoring", "anomaly_detection", "impulse_event"],
        speech_follow_up_tasks=["asr"],
        non_speech_follow_up_tasks=["esc"],
        high_risk_follow_up_tasks=["emotion"],
        strict_privacy=True,
        allow_raw_audio_export=False,
        max_parallel_tasks=4,
    ),
    "low_power": AgentPolicy(
        name="low_power",
        always_on_tasks=["vad", "audio_quality_monitoring", "impulse_event"],
        speech_follow_up_tasks=["asr"],
        non_speech_follow_up_tasks=["anomaly_detection"],
        high_risk_follow_up_tasks=["emotion", "esc"],
        strict_privacy=True,
        max_parallel_tasks=3,
    ),
}


def get_policy(profile: str) -> AgentPolicy:
    """Return a policy preset, falling back to balanced for unknown names."""
    return PROFILE_POLICIES.get((profile or "balanced").lower(), PROFILE_POLICIES["balanced"])
