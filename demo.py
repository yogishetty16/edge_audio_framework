import json
import numpy as np
from agent import AudioAgent
from fast_run import print_agent_decision

def run_demo():
    print("="*60)
    print("  EDGE AUDIO FRAMEWORK AGENTIC LAYER DEMONSTRATION")
    print("="*60)

    # Initialize AudioAgent
    agent = AudioAgent(profile="balanced")
    
    # -------------------------------------------------------------
    # SETUP: Calibrate the environment on dummy silence
    # -------------------------------------------------------------
    print("\n--- [STAGE 0] RUNNING STARTUP CALIBRATION ---")
    dummy_silence = np.random.normal(0, 0.005, 16000 * 2) # 2s of quiet noise
    report = agent.run_calibration(dummy_silence, 16000)
    print("Calibration Baseline & Thresholds:")
    print(json.dumps(report, indent=2))

    # -------------------------------------------------------------
    # Scenario 1 — Normal ambient audio
    # -------------------------------------------------------------
    print("\n--- [SCENARIO 1] NORMAL AMBIENT AUDIO ---")
    scenario_1_results = {
        "vad": {"is_speech": False, "speech_ratio": 0.02, "speech_duration_sec": 0.1, "success": True},
        "audio_quality_monitoring": {"quality_label": "good", "snr_db": 22.0, "mos": 4.1, "success": True},
        "anomaly_detection": {"is_anomaly": False, "anomaly_score": -0.15, "success": True},
        "esc": {"top_class": "silence", "top_score": 0.91, "success": True}
    }
    decision_1 = agent.decide(scenario_1_results)
    print_agent_decision(decision_1)

    # -------------------------------------------------------------
    # Scenario 2 — Worker distress (triggers investigation)
    # -------------------------------------------------------------
    print("\n--- [SCENARIO 2] WORKER DISTRESS (TRIGGERS INVESTIGATION) ---")
    scenario_2_results = {
        "vad": {"is_speech": True, "speech_ratio": 0.89, "speech_duration_sec": 4.5, "success": True},
        "emotion": {"top_emotion": "angry", "top_score": 0.81, "success": True},
        "asr": {"text": "help me there is a fire", "language": "en", "confidence": 0.94, "success": True},
        "keyword_spotting": {"top_label": "help", "top_score": 0.88, "success": True},
        "esc": {"top_class": "scream", "top_score": 0.74, "success": True},
        "audio_quality_monitoring": {"quality_label": "good", "snr_db": 18.0, "mos": 3.7, "success": True},
        "anomaly_detection": {"is_anomaly": True, "anomaly_score": -0.61, "success": True},
        "speaker_id": {"num_speakers": 1, "success": True}
    }
    decision_2 = agent.decide(scenario_2_results)
    print_agent_decision(decision_2)

    investigation_id = decision_2.metadata.get("investigation_id")
    print(f"\nInvestigation opened: {investigation_id}")

    # -------------------------------------------------------------
    # Scenario 3 — Follow-up confirms threat (investigation verdict)
    # -------------------------------------------------------------
    print("\n--- [SCENARIO 3] FOLLOW-UP INVESTIGATION CYCLE & VERDICT ---")
    # Follow-up 1 (+10s): High emotion, speech present -> Threat present
    print("\n[SCENARIO 3] Simulating Follow-up 1 (+10s)...")
    fu1_results = {
        "vad": {"is_speech": True, "speech_ratio": 0.70, "speech_duration_sec": 3.5, "success": True},
        "emotion": {"top_emotion": "angry", "top_score": 0.65, "success": True},
        "audio_quality_monitoring": {"quality_label": "good", "snr_db": 19.0, "mos": 3.8, "success": True},
        "anomaly_detection": {"is_anomaly": True, "anomaly_score": -0.50, "success": True}
    }
    dec_fu1 = agent._synthesis.synthesize(
        fu1_results, {}, agent._policy_name, thresholds=agent.calibration_agent.get_thresholds()
    )
    # Convert dict returned by synthesize to AgentDecision
    from agent.audio_agent import _dict_to_decision
    dec_fu1_obj = _dict_to_decision(dec_fu1)
    agent.investigation_agent.record_followup(
        investigation_id, 0, fu1_results, dec_fu1_obj
    )
    print("Follow-up 1 registered.")

    # Follow-up 2 (+30s): Lower emotion, speech fading -> Threat still present (confirmed count += 1)
    print("\n[SCENARIO 3] Simulating Follow-up 2 (+30s)...")
    fu2_results = {
        "vad": {"is_speech": True, "speech_ratio": 0.40, "speech_duration_sec": 2.0, "success": True},
        "emotion": {"top_emotion": "neutral", "top_score": 0.50, "success": True},
        "audio_quality_monitoring": {"quality_label": "good", "snr_db": 20.0, "mos": 4.0, "success": True},
        "anomaly_detection": {"is_anomaly": True, "anomaly_score": -0.48, "success": True}
    }
    dec_fu2 = agent._synthesis.synthesize(
        fu2_results, {}, agent._policy_name, thresholds=agent.calibration_agent.get_thresholds()
    )
    dec_fu2_obj = _dict_to_decision(dec_fu2)
    agent.investigation_agent.record_followup(
        investigation_id, 1, fu2_results, dec_fu2_obj
    )
    print("Follow-up 2 registered.")

    # Follow-up 3 (+60s): Normal ambient noise -> Threat resolved (Total confirmed: 2/3, reaches THREAT_CONFIRMED verdict)
    print("\n[SCENARIO 3] Simulating Follow-up 3 (+60s)...")
    fu3_results = {
        "vad": {"is_speech": False, "speech_ratio": 0.01, "speech_duration_sec": 0.05, "success": True},
        "audio_quality_monitoring": {"quality_label": "good", "snr_db": 22.0, "mos": 4.2, "success": True},
        "anomaly_detection": {"is_anomaly": False, "anomaly_score": -0.05, "success": True}
    }
    dec_fu3 = agent._synthesis.synthesize(
        fu3_results, {}, agent._policy_name, thresholds=agent.calibration_agent.get_thresholds()
    )
    dec_fu3_obj = _dict_to_decision(dec_fu3)
    
    # This 3rd recording will trigger _reach_verdict() and close the investigation
    print("Recording final follow-up...")
    agent.investigation_agent.record_followup(
        investigation_id, 2, fu3_results, dec_fu3_obj
    )
    
    # Fetch final report to display
    final_report = agent.investigation_agent.get_investigation_report(investigation_id)
    print("\nFull Closed Investigation Verdict Report:")
    print(json.dumps(final_report, indent=2))

    # -------------------------------------------------------------
    # Scenario 4 — Watchdog shift summary (10 events)
    # -------------------------------------------------------------
    print("\n--- [SCENARIO 4] WATCHDOG SHIFT SUMMARY REPORT ---")
    print("Running 10 sequential events to trigger the periodic watchdog analysis loop...")
    
    # Reset watchdog count if needed, or let it accumulate
    normal_results = {
        "vad": {"is_speech": False, "speech_ratio": 0.01, "success": True},
        "audio_quality_monitoring": {"quality_label": "good", "snr_db": 20.0, "mos": 4.0, "success": True},
        "anomaly_detection": {"is_anomaly": False, "anomaly_score": -0.05, "success": True}
    }
    risk_results = {
        "vad": {"is_speech": False, "speech_ratio": 0.00, "success": True},
        "audio_quality_monitoring": {"quality_label": "fair", "snr_db": 11.0, "mos": 3.0, "success": True},
        "anomaly_detection": {"is_anomaly": True, "anomaly_score": -0.45, "success": True},
        "esc": {"top_class": "explosion", "top_score": 0.60, "success": True}
    }
    
    # Run 10 events:
    # 1-4: normal_audio
    # 5-7: acoustic_risk_event (using risk_results)
    # 8-10: normal_audio (Event 10 will trigger the Watchdog analysis report)
    for idx in range(1, 11):
        if 5 <= idx <= 7:
            res = risk_results
        else:
            res = normal_results
        decision = agent.decide(res)
        print(f"  Event {idx}/10 processed. Event Type: {decision.event_type}, Priority: {decision.priority}")
        
        # Check if watchdog report was generated on Event 10
        if idx == 10:
            report_data = decision.metadata.get("watchdog_report")
            print(f"\n[WATCHDOG REPORT GENERATED ON EVENT {idx}]")
            print(json.dumps(report_data, indent=2))

    print("\n" + "="*60)
    print("=== DEMO COMPLETE ===")
    print("Framework demonstrated: calibration, investigation loop, watchdog analysis")
    print("All 3 agentic behaviours shown without microphone input")
    print("="*60)

if __name__ == "__main__":
    run_demo()
