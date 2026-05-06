"""
core/pipeline.py
================
Sequential audio analysis pipeline.
Runs each task in order, passing the AudioData object to each module's
analyze() function and collecting results as dicts.
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .audio_io import AudioData

logger = logging.getLogger(__name__)

# Map public task names → Python module paths
_TASK_MODULE_MAP: Dict[str, str] = {
    "vad":                      "tasks.vad",
    "asr":                      "tasks.asr",
    "keyword_spotting":         "tasks.keyword_spotting",
    "speaker_id":               "tasks.speaker_id",
    "emotion":                  "tasks.emotion",
    "speech_quality":           "tasks.speech_quality",
    "lang_accent_id":           "tasks.accent_lang_id",
    "esc":                      "tasks.esc",
    "acoustic_event_detection": "tasks.acoustic_event_detection",
    "audio_quality_monitoring": "tasks.audio_quality_monitor",
    "music_speech_detection":   "tasks.music_speech_detection",
    "anomaly_detection":        "tasks.anomaly_detection",
    "music_genre":              "tasks.music_genre",
}


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert a result object (dataclass or dict) to a plain dict."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    # dataclass
    if hasattr(result, "__dataclass_fields__"):
        try:
            return asdict(result)
        except Exception:
            pass
    # fallback
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    return {"value": str(result)}


class AudioPipeline:
    """
    Sequential pipeline that runs a list of analysis tasks on AudioData.

    Parameters
    ----------
    tasks : List of task name strings, e.g. ["vad", "asr", "emotion"]
    """

    def __init__(self, tasks: List[str]) -> None:
        self.tasks = tasks

    def run(self, audio: AudioData, active_tasks: Optional[List[str]] = None, **kwargs) -> Dict[str, Dict]:
        """
        Run every task sequentially.

        Returns a dict mapping task_name → result_dict.
        Each result_dict always has:
          - "success" (bool)
          - "latency_ms" (float)
          - task-specific keys
        """
        results: Dict[str, Dict] = {}
        tasks_to_run = active_tasks if active_tasks is not None else self.tasks

        for task_name in tasks_to_run:
            module_path = _TASK_MODULE_MAP.get(task_name)
            if module_path is None:
                logger.warning(f"[pipeline] Unknown task '{task_name}' — skipped.")
                results[task_name] = {
                    "success": False,
                    "error": f"Unknown task '{task_name}'",
                }
                continue

            try:
                mod = importlib.import_module(module_path)
                t0 = time.perf_counter()
                raw = mod.analyze(audio, **kwargs)
                latency_ms = (time.perf_counter() - t0) * 1000

                d = _result_to_dict(raw)
                d.setdefault("success", True)
                d["latency_ms"] = round(latency_ms, 2)
                results[task_name] = d

            except Exception as exc:
                logger.error(f"[pipeline] Task '{task_name}' failed: {exc}", exc_info=True)
                results[task_name] = {
                    "success": False,
                    "error": str(exc),
                    "latency_ms": 0.0,
                }

        return results
