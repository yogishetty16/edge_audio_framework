import unittest
from tempfile import TemporaryDirectory

from agent import AudioAgent, AudioMemory


class AudioAgentTests(unittest.TestCase):
    def test_high_risk_incident_from_multiple_signals(self):
        agent = AudioAgent("industrial_safety")

        decision = agent.decide({
            "vad": {"success": True, "speech_ratio": 0.72, "total_speech_sec": 4.1},
            "audio_quality_monitoring": {
                "success": True,
                "quality_label": "fair",
                "snr_db": 11.2,
            },
            "anomaly_detection": {
                "success": True,
                "is_anomaly": True,
                "anomaly_score": -0.22,
            },
            "emotion": {
                "success": True,
                "top_emotion": "angry",
                "top_score": 0.71,
            },
            "esc": {
                "success": True,
                "top_class": "Alarm",
                "top_score": 0.44,
            },
        })

        self.assertEqual(decision.event_type, "possible_safety_incident")
        self.assertEqual(decision.priority, "high")
        self.assertEqual(decision.privacy_mode, "metadata_plus_redacted_evidence")
        self.assertIn("audio_anomaly", decision.risk_signals)
        self.assertIn("stressed_speech", decision.risk_signals)
        self.assertFalse(decision.data_export["raw_audio"])

    def test_orchestrated_plan_changes_for_non_speech(self):
        agent = AudioAgent("balanced")

        initial = agent.plan_initial_tasks()
        self.assertIn("vad", initial.tasks)
        self.assertIn("audio_quality_monitoring", initial.tasks)

        follow_up = agent.plan_follow_up_tasks({
            "vad": {"success": True, "speech_ratio": 0.0},
            "audio_quality_monitoring": {"success": True, "quality_label": "good", "snr_db": 18},
        }, already_run=initial.tasks)

        self.assertIn("esc", follow_up.tasks)
        self.assertNotIn("asr", follow_up.tasks)

    def test_privacy_first_does_not_export_raw_audio_for_normal_speech(self):
        agent = AudioAgent("privacy_first")

        decision = agent.decide({
            "vad": {"success": True, "speech_ratio": 0.4},
            "asr": {"success": True, "text": "start inspection checklist", "language": "en"},
            "audio_quality_monitoring": {"success": True, "quality_label": "good", "snr_db": 20},
        })

        self.assertEqual(decision.event_type, "speech_workflow_event")
        self.assertFalse(decision.data_export["raw_audio"])
        self.assertTrue(decision.data_export["transcript"])

    def test_speech_classifier_overrides_vad_and_downgrades_isolated_anomaly(self):
        agent = AudioAgent("industrial_safety")

        first_pass = {
            "vad": {"success": True, "speech_ratio": 0.0},
            "audio_quality_monitoring": {"success": True, "quality_label": "good", "snr_db": 28.8},
            "anomaly_detection": {"success": True, "is_anomaly": True, "anomaly_score": -0.31},
            "esc": {"success": True, "top_class": "Speech", "top_score": 0.47},
        }
        follow_up = agent.plan_follow_up_tasks(first_pass, already_run=first_pass.keys())
        decision = agent.decide(first_pass)

        self.assertIn("asr", follow_up.tasks)
        self.assertIn("emotion", follow_up.tasks)
        self.assertNotIn("audio_anomaly", decision.risk_signals)
        self.assertEqual(decision.event_type, "speech_detected")

    def test_generic_keyword_does_not_create_alert_risk(self):
        agent = AudioAgent("industrial_safety")

        decision = agent.decide({
            "vad": {"success": True, "speech_ratio": 0.0},
            "audio_quality_monitoring": {"success": True, "quality_label": "excellent", "snr_db": 35.8},
            "anomaly_detection": {"success": True, "is_anomaly": False, "anomaly_score": 0.0},
            "esc": {"success": True, "top_class": "Speech", "top_score": 0.77},
            "asr": {"success": True, "text": "for the ideas you never acted on", "language": "en"},
            "emotion": {"success": True, "top_emotion": "angry", "top_score": 0.75},
            "keyword_spotting": {"success": True, "top_label": "right", "top_score": 0.65},
        })

        self.assertNotIn("alert_keyword", decision.risk_signals)
        self.assertIn("stressed_speech", decision.risk_signals)
        self.assertNotEqual(decision.event_type, "possible_safety_incident")

    def test_explicit_alert_keyword_still_raises_risk(self):
        agent = AudioAgent("industrial_safety")

        decision = agent.decide({
            "vad": {"success": True, "speech_ratio": 0.4},
            "audio_quality_monitoring": {"success": True, "quality_label": "good", "snr_db": 18.0},
            "emotion": {"success": True, "top_emotion": "angry", "top_score": 0.7},
            "keyword_spotting": {"success": True, "top_label": "stop", "top_score": 0.7},
        })

        self.assertIn("alert_keyword", decision.risk_signals)
        self.assertEqual(decision.event_type, "possible_safety_incident")

    def test_memory_persists_and_escalates_related_risk(self):
        agent = AudioAgent("industrial_safety")

        with TemporaryDirectory() as tmp:
            memory = AudioMemory(f"{tmp}/events.jsonl")
            first = agent.decide({
                "vad": {"success": True, "speech_ratio": 0.5},
                "emotion": {"success": True, "top_emotion": "angry", "top_score": 0.7},
            })
            memory.append_decision(first, source="test")

            second = agent.decide({
                "vad": {"success": True, "speech_ratio": 0.4},
                "emotion": {"success": True, "top_emotion": "angry", "top_score": 0.72},
            })
            context = memory.build_context(second)
            memory.enrich_decision(second, context)

            self.assertGreaterEqual(context["recent_count"], 1)
            self.assertIn("timeline_escalation", second.risk_signals)
            self.assertEqual(second.priority, "high")

    def test_memory_does_not_escalate_unrelated_risk(self):
        agent = AudioAgent("balanced")

        with TemporaryDirectory() as tmp:
            memory = AudioMemory(f"{tmp}/events.jsonl")
            old = agent.decide({
                "vad": {"success": True, "speech_ratio": 0.5},
                "asr": {"success": True, "text": "normal speech"},
            })
            old.priority = "high"
            old.event_type = "possible_safety_incident"
            old.risk_signals = ["stressed_speech"]
            memory.append_decision(old, source="test")

            current = agent.decide({
                "vad": {"success": True, "speech_ratio": 0.0},
                "anomaly_detection": {"success": True, "is_anomaly": True, "anomaly_score": -0.2},
            })
            context = memory.build_context(current)
            memory.enrich_decision(current, context)

            self.assertEqual(context["related_count"], 0)
            self.assertNotIn("timeline_escalation", current.risk_signals)

    def test_impulse_event_creates_specific_high_priority_decision(self):
        agent = AudioAgent("balanced")

        decision = agent.decide({
            "vad": {"success": True, "speech_ratio": 0.0},
            "audio_quality_monitoring": {"success": True, "quality_label": "good", "snr_db": 15.0},
            "impulse_event": {
                "success": True,
                "is_impulse": True,
                "event_label": "gunshot_or_impact_like_impulse",
                "confidence": 0.77,
            },
        })

        self.assertEqual(decision.event_type, "possible_impulse_threat")
        self.assertEqual(decision.priority, "high")
        self.assertIn("impulse_event", decision.risk_signals)


if __name__ == "__main__":
    unittest.main()
