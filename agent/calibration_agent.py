import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class CalibrationAgent:
    def __init__(self, calibration_path="agent_memory/calibration.json"):
        self.calibration_path = Path(calibration_path)
        self.is_calibrated = False
        self.baseline = {}
        self.thresholds = {}
        self.environment_type = "unknown"
        self.adjustments_count = 0
        self.false_positive_tracker = defaultdict(list)
        
        # Load existing calibration if it exists
        if self._load_calibration():
            self.is_calibrated = True

    def calibrate(self, waveform, sr) -> dict:
        """Run tasks directly on a short startup waveform and set thresholds."""
        try:
            from core.audio_io import AudioData
            audio = AudioData(waveform=waveform, sample_rate=sr, duration_sec=len(waveform)/sr, n_channels=1)
            
            # Import tasks directly
            import tasks.vad as vad_task
            import tasks.audio_quality_monitor as aq_task
            import tasks.anomaly_detection as anomaly_task
            import tasks.esc as esc_task
            
            # Run tasks safely
            try:
                vad_res = vad_task.analyze(audio)
                baseline_speech_ratio = getattr(vad_res, "speech_ratio", 0.0)
            except Exception as e:
                logger.warning("VAD failed during calibration: %s", e)
                baseline_speech_ratio = 0.0

            try:
                aq_res = aq_task.analyze(audio)
                baseline_snr_db = getattr(aq_res, "snr_db", 15.0)
            except Exception as e:
                logger.warning("Audio quality failed during calibration: %s", e)
                baseline_snr_db = 15.0

            try:
                anomaly_res = anomaly_task.analyze(audio)
                baseline_anomaly_score = getattr(anomaly_res, "anomaly_score", 0.0)
            except Exception as e:
                logger.warning("Anomaly detection failed during calibration: %s", e)
                baseline_anomaly_score = 0.0

            try:
                esc_res = esc_task.analyze(audio)
                baseline_noise_class = getattr(esc_res, "top_class", "silence")
            except Exception as e:
                logger.warning("ESC failed during calibration: %s", e)
                baseline_noise_class = "silence"

            # Compute RMS energy in dB
            rms = float(np.sqrt(np.mean(waveform ** 2))) if len(waveform) > 0 else 0.0
            baseline_energy_db = float(20 * np.log10(rms + 1e-9))

            # Environment Classification
            noise_lower = str(baseline_noise_class).lower()
            if any(x in noise_lower for x in ["machinery", "engine", "factory", "industrial", "drill", "compressor", "ventilation"]):
                environment_type = "machinery_heavy"
            elif any(x in noise_lower for x in ["outdoor", "nature", "rain", "wind", "water", "traffic", "street", "car", "birds"]):
                environment_type = "outdoor"
            elif baseline_energy_db < -45.0 and baseline_snr_db > 15.0:
                environment_type = "silent_room"
            elif baseline_energy_db < -30.0 and baseline_snr_db >= 10.0:
                environment_type = "office"
            else:
                environment_type = "industrial"

            self.environment_type = environment_type

            # Calculate Thresholds based on environment
            if environment_type == "silent_room":
                speech_ratio_threshold = 0.15
                anomaly_score_threshold = -0.30
                snr_alert_threshold = 12.0
                energy_spike_threshold = baseline_energy_db + 15
            elif environment_type == "office":
                speech_ratio_threshold = 0.25
                anomaly_score_threshold = -0.40
                snr_alert_threshold = 8.0
                energy_spike_threshold = baseline_energy_db + 12
            elif environment_type in ("industrial", "machinery_heavy"):
                speech_ratio_threshold = 0.40
                anomaly_score_threshold = -0.55
                snr_alert_threshold = 4.0
                energy_spike_threshold = baseline_energy_db + 20
            else:  # outdoor
                speech_ratio_threshold = 0.30
                anomaly_score_threshold = -0.45
                snr_alert_threshold = 6.0
                energy_spike_threshold = baseline_energy_db + 18

            self.baseline = {
                "baseline_speech_ratio": float(baseline_speech_ratio),
                "baseline_snr_db": float(baseline_snr_db),
                "baseline_anomaly_score": float(baseline_anomaly_score),
                "baseline_noise_class": str(baseline_noise_class),
                "baseline_energy_db": float(baseline_energy_db)
            }

            self.thresholds = {
                "speech_ratio_threshold": float(speech_ratio_threshold),
                "anomaly_score_threshold": float(anomaly_score_threshold),
                "snr_alert_threshold": float(snr_alert_threshold),
                "energy_spike_threshold": float(energy_spike_threshold)
            }

            self.is_calibrated = True
            self.adjustments_count = 0
            self._save_calibration()

            summary = (
                f"Calibrated for {environment_type} environment. "
                f"Baseline SNR: {baseline_snr_db:.1f}dB, ambient energy: {baseline_energy_db:.1f}dB. "
                f"Thresholds adjusted to optimize detection sensitivity."
            )

            return {
                "calibrated_at": datetime.now(timezone.utc).isoformat(),
                "environment_type": environment_type,
                "baseline": self.baseline,
                "thresholds": self.thresholds,
                "calibration_version": 1,
                "summary": summary
            }

        except Exception as e:
            logger.error("Calibration process failed: %s", e)
            return {"error": str(e), "summary": "Calibration failed due to internal error."}

    def update_thresholds(self, decision: dict) -> dict | None:
        """Track false positives and tune thresholds dynamically."""
        if not self.is_calibrated:
            return None

        try:
            event_type = decision.get("event_type", "normal_audio")
            confidence = decision.get("confidence", 0.0)
            is_fp = (event_type != "normal_audio") and (confidence < 0.45)
            
            risk_signals = decision.get("risk_signals", [])
            if not risk_signals:
                return None

            changed = {}
            signal_map = {
                "stressed_speech": "speech_ratio_threshold",
                "alert_keyword": "speech_ratio_threshold",
                "transcript_alert": "speech_ratio_threshold",
                "audio_anomaly": "anomaly_score_threshold",
                "alert_sound": "anomaly_score_threshold",
                "low_audio_quality": "snr_alert_threshold",
                "impulse_event": "energy_spike_threshold"
            }

            for sig in risk_signals:
                if sig not in signal_map:
                    continue
                tracker = self.false_positive_tracker[sig]
                tracker.append((confidence, is_fp))
                if len(tracker) > 20:
                    tracker.pop(0)

                # Need at least 5 events to make statistical adjustments
                if len(tracker) >= 5:
                    fp_rate = sum(1 for _, fp in tracker if fp) / len(tracker)
                    target_threshold = signal_map[sig]
                    current_val = self.thresholds.get(target_threshold)
                    
                    if current_val is None:
                        continue

                    # FP rate too high: make less sensitive
                    if fp_rate > 0.40:
                        if target_threshold == "speech_ratio_threshold":
                            new_val = min(0.95, current_val * 1.05)
                        elif target_threshold == "anomaly_score_threshold":
                            new_val = max(-0.95, current_val * 1.05) # e.g. -0.40 -> -0.42 (more negative is less sensitive)
                        elif target_threshold == "snr_alert_threshold":
                            new_val = max(1.0, current_val * 0.95) # lower SNR limit triggers less easily
                        elif target_threshold == "energy_spike_threshold":
                            new_val = current_val + 1.0 if current_val < 0 else current_val * 1.05
                        
                        if new_val != current_val:
                            self.thresholds[target_threshold] = round(new_val, 4)
                            changed[target_threshold] = round(new_val, 4)
                            self.adjustments_count += 1
                            logger.info(f"[CALIBRATION] Threshold for {sig} raised (less sensitive): {current_val} -> {new_val:.4f} (fp_rate={fp_rate:.0%})")

                    # FP rate very low: make more sensitive
                    elif fp_rate < 0.05:
                        if target_threshold == "speech_ratio_threshold":
                            new_val = max(0.02, current_val * 0.97)
                        elif target_threshold == "anomaly_score_threshold":
                            new_val = min(-0.02, current_val * 0.97) # e.g. -0.40 -> -0.388 (closer to 0 is more sensitive)
                        elif target_threshold == "snr_alert_threshold":
                            new_val = min(25.0, current_val * 1.03) # higher SNR limit triggers more easily
                        elif target_threshold == "energy_spike_threshold":
                            new_val = current_val - 0.5 if current_val < 0 else current_val * 0.97
                        
                        if new_val != current_val:
                            self.thresholds[target_threshold] = round(new_val, 4)
                            changed[target_threshold] = round(new_val, 4)
                            self.adjustments_count += 1
                            logger.info(f"[CALIBRATION] Threshold for {sig} lowered (more sensitive): {current_val} -> {new_val:.4f} (fp_rate={fp_rate:.0%})")

            if changed:
                self._save_calibration()
                return changed
            return None

        except Exception as e:
            logger.warning("Error updating thresholds: %s", e)
            return None

    def get_thresholds(self) -> dict:
        if self.is_calibrated and self.thresholds:
            return self.thresholds
        return {
            "speech_ratio_threshold": 0.30,
            "anomaly_score_threshold": -0.40,
            "snr_alert_threshold": 8.0,
            "energy_spike_threshold": -10.0
        }

    def get_calibration_summary(self) -> str:
        if not self.is_calibrated:
            return "Running on default thresholds. Run calibration for environment-specific optimization."
        
        t_str = " | ".join(f"{k.split('_')[0]}={v:.2f}" for k, v in self.thresholds.items())
        return (
            f"Calibrated for {self.environment_type} environment. "
            f"Current thresholds: {t_str}. "
            f"Made {self.adjustments_count} adjustments to thresholds since calibration."
        )

    def _load_calibration(self) -> bool:
        try:
            if not self.calibration_path.exists():
                return False
            with open(self.calibration_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.baseline = data.get("baseline", {})
            self.thresholds = data.get("thresholds", {})
            self.environment_type = data.get("environment_type", "unknown")
            self.adjustments_count = data.get("adjustments_count", 0)
            return True
        except Exception as e:
            logger.warning("Failed to load calibration: %s", e)
            return False

    def _save_calibration(self):
        try:
            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            cal_data = {
                "calibrated_at": datetime.now(timezone.utc).isoformat(),
                "environment_type": self.environment_type,
                "baseline": self.baseline,
                "thresholds": self.thresholds,
                "calibration_version": 1,
                "adjustments_count": self.adjustments_count
            }
            with open(self.calibration_path, "w", encoding="utf-8") as f:
                json.dump(cal_data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save calibration: %s", e)
