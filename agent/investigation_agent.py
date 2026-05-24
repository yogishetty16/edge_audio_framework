import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

class InvestigationAgent:
    def __init__(self, memory_path="agent_memory/investigations.jsonl"):
        self.memory_path = Path(memory_path)
        self.active_investigations = {}
        
        # Ensure file exists
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.touch()

        # Load any incomplete investigations from file on startup
        self._load_active_investigations()

    def should_investigate(self, decision: dict) -> bool:
        """Determines if a decision qualifies for starting a new investigation."""
        try:
            priority = str(decision.get("priority", "low")).lower()
            event_type = decision.get("event_type", "")
            
            # Must be HIGH priority and one of the critical safety/acoustic categories
            if priority != "high" or event_type not in (
                "possible_safety_incident",
                "possible_impulse_threat",
                "acoustic_risk_event"
            ):
                return False

            # Check if there is already an active investigation for the same event type in the last 120s
            now = datetime.now(timezone.utc)
            for inv in self.active_investigations.values():
                trigger_event = inv.get("triggered_by", {})
                if trigger_event.get("event_type") == event_type:
                    triggered_at_str = inv.get("triggered_at")
                    if triggered_at_str:
                        try:
                            triggered_at = datetime.fromisoformat(triggered_at_str)
                            if (now - triggered_at).total_seconds() < 120:
                                return False
                        except Exception:
                            pass
            return True
        except Exception as e:
            logger.warning("Error in should_investigate: %s", e)
            return False

    def open_investigation(self, decision: dict) -> str:
        """Opens a new investigation for a HIGH priority event."""
        try:
            investigation_id = str(uuid.uuid4())
            now_iso = datetime.now(timezone.utc).isoformat()
            
            # Structure follow-ups
            follow_ups = [
                {
                    "scheduled_at_offset_seconds": offset,
                    "recorded_at": None,
                    "task_results": None,
                    "decision": None,
                    "comparison": None
                }
                for offset in (10, 30, 60)
            ]

            inv_data = {
                "investigation_id": investigation_id,
                "triggered_by": decision,
                "triggered_at": now_iso,
                "status": "active",
                "follow_ups": follow_ups,
                "verdict": None,
                "verdict_reasoning": "",
                "timeline_summary": "",
                "completed_at": None
            }

            self.active_investigations[investigation_id] = inv_data
            self._save_investigation(inv_data)
            
            short_id = investigation_id[:8]
            print(f"[INVESTIGATION] Opened #{short_id} — {decision.get('event_type')} at {decision.get('priority').upper()}. Follow-ups scheduled at +10s, +30s, +60s.")
            
            return investigation_id
        except Exception as e:
            logger.error("Failed to open investigation: %s", e)
            return ""

    def check_due_followups(self) -> list[dict]:
        """Scans active investigations and returns follow-ups that are due."""
        try:
            now = datetime.now(timezone.utc)
            due = []
            for iid, inv in self.active_investigations.items():
                triggered_at_str = inv.get("triggered_at")
                if not triggered_at_str:
                    continue
                try:
                    triggered_at = datetime.fromisoformat(triggered_at_str)
                except Exception:
                    continue

                for idx, fu in enumerate(inv.get("follow_ups", [])):
                    if fu.get("recorded_at") is None:
                        offset = fu.get("scheduled_at_offset_seconds", 0)
                        due_time = triggered_at.timestamp() + offset
                        if now.timestamp() >= due_time:
                            due.append({
                                "investigation_id": iid,
                                "follow_up_index": idx,
                                "offset": offset
                            })
            return due
        except Exception as e:
            logger.warning("Error checking due followups: %s", e)
            return []

    def record_followup(self, investigation_id: str, follow_up_index: int, task_results: dict, decision: dict) -> dict:
        """Records a follow-up event's results and updates the investigation state."""
        try:
            inv = self.active_investigations.get(investigation_id)
            if not inv:
                raise ValueError(f"Investigation {investigation_id} not found in active list.")

            fu = inv["follow_ups"][follow_up_index]
            fu["recorded_at"] = datetime.now(timezone.utc).isoformat()
            fu["task_results"] = task_results
            # Store serialisable dict format of decision
            fu["decision"] = decision.to_dict() if hasattr(decision, "to_dict") else decision
            
            # Compare to trigger
            fu["comparison"] = self._compare_to_trigger(investigation_id, follow_up_index)
            
            # Check if all completed
            all_done = all(f.get("recorded_at") is not None for f in inv["follow_ups"])
            if all_done:
                self._reach_verdict(investigation_id)
            else:
                self._save_investigation(inv)

            return fu
        except Exception as e:
            logger.error("Failed to record follow-up: %s", e)
            return {}

    def get_active_summary(self) -> list[str]:
        """Returns active investigation summaries."""
        try:
            summaries = []
            now = datetime.now(timezone.utc)
            for iid, inv in self.active_investigations.items():
                short_id = iid[:8]
                event = inv["triggered_by"].get("event_type", "unknown")
                triggered_at_str = inv.get("triggered_at")
                age_str = "unknown age"
                if triggered_at_str:
                    try:
                        triggered_at = datetime.fromisoformat(triggered_at_str)
                        age = int((now - triggered_at).total_seconds())
                        age_str = f"{age}s ago"
                    except Exception:
                        pass
                
                done = sum(1 for f in inv["follow_ups"] if f.get("recorded_at") is not None)
                summaries.append(f"#{short_id} — {event} — opened {age_str} — {done}/3 follow-ups complete")
            return summaries
        except Exception as e:
            logger.warning("Error getting active summary: %s", e)
            return []

    def get_investigation_report(self, investigation_id: str) -> dict | None:
        """Reads investigations.jsonl to find the full historical status of an investigation."""
        try:
            if not self.memory_path.exists():
                return None
            target_inv = None
            with open(self.memory_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("investigation_id") == investigation_id:
                            target_inv = data
                    except Exception:
                        continue
            return target_inv
        except Exception as e:
            logger.warning("Failed to retrieve investigation report: %s", e)
            return None

    def _compare_to_trigger(self, investigation_id: str, follow_up_index: int) -> dict:
        """Compares a follow-up's decision to the initial trigger decision."""
        inv = self.active_investigations[investigation_id]
        trigger = inv["triggered_by"]
        follow_up = inv["follow_ups"][follow_up_index]["decision"]

        t_type = trigger.get("event_type", "")
        f_type = follow_up.get("event_type", "")
        
        t_priority = trigger.get("priority", "low").lower()
        f_priority = follow_up.get("priority", "low").lower()
        
        priority_map = {"low": 1, "medium": 2, "high": 3}
        t_val = priority_map.get(t_priority, 1)
        f_val = priority_map.get(f_priority, 1)
        
        if f_val > t_val:
            priority_change = "escalated"
        elif f_val < t_val:
            priority_change = "de-escalated"
        else:
            priority_change = "same"

        t_risks = set(trigger.get("risk_signals", []))
        f_risks = set(follow_up.get("risk_signals", []))
        
        shared = list(t_risks.intersection(f_risks))
        new_risks = list(f_risks.difference(t_risks))
        resolved = list(t_risks.difference(f_risks))
        
        t_conf = trigger.get("confidence", 0.0)
        f_conf = follow_up.get("confidence", 0.0)
        
        # Threat is present if follow-up is medium/high AND shares at least 1 risk signal
        threat_present = (f_priority in ("medium", "high")) and (len(shared) > 0)

        return {
            "event_type_match": t_type == f_type,
            "priority_change": priority_change,
            "shared_risk_signals": shared,
            "new_risk_signals": new_risks,
            "resolved_risk_signals": resolved,
            "confidence_delta": round(f_conf - t_conf, 4),
            "threat_still_present": threat_present
        }

    def _reach_verdict(self, investigation_id: str) -> dict:
        """Aggregates all follow-ups to reach a final threat verdict and close the investigation."""
        inv = self.active_investigations[investigation_id]
        follow_ups = inv["follow_ups"]
        trigger = inv["triggered_by"]
        
        confirmed_count = sum(1 for f in follow_ups if f["comparison"].get("threat_still_present") is True)
        
        all_shared = set()
        for f in follow_ups:
            all_shared.update(f["comparison"].get("shared_risk_signals", []))
        shared_signals = list(all_shared)

        if confirmed_count >= 2:
            verdict = "threat_confirmed"
            verdict_reasoning = f"Threat signal persisted across {confirmed_count}/3 follow-up windows. Risk signals {shared_signals} remained active throughout."
        elif confirmed_count == 1:
            verdict = "threat_resolved"
            verdict_reasoning = "Initial threat signal detected but did not persist. Only 1 of 3 follow-up windows showed continued activity. Likely a transient event."
        else: # confirmed_count == 0
            first_conf = trigger.get("confidence", 0.0)
            if first_conf < 0.55:
                verdict = "false_alarm"
                verdict_reasoning = f"No follow-up windows confirmed the initial detection. Original confidence was low ({first_conf:.0%}). Likely a false positive — threshold will be adjusted."
            else:
                verdict = "false_alarm"
                verdict_reasoning = f"No follow-up windows confirmed the initial detection despite high initial confidence ({first_conf:.0%}). Isolated acoustic event."

        # Build timeline summary
        summaries = []
        for f in follow_ups:
            offset = f["scheduled_at_offset_seconds"]
            present = f["comparison"].get("threat_still_present")
            conf = f["decision"].get("confidence", 0.0)
            if present:
                summaries.append(f"+{offset}s: threat present (confidence {conf*100:.0f}%)")
            else:
                summaries.append(f"+{offset}s: threat cleared")

        timeline = ", ".join(summaries)
        t_time = trigger.get("timestamp", datetime.now(timezone.utc).isoformat())
        timeline_summary = f"HIGH event detected at {t_time}. Follow-up timeline: {timeline}. Final verdict: {verdict}."

        inv["verdict"] = verdict
        inv["verdict_reasoning"] = verdict_reasoning
        inv["timeline_summary"] = timeline_summary
        inv["status"] = "completed"
        inv["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        # Calculate duration
        try:
            t_dt = datetime.fromisoformat(t_time)
            c_dt = datetime.fromisoformat(inv["completed_at"])
            duration = int((c_dt - t_dt).total_seconds())
        except Exception:
            duration = 60

        short_id = investigation_id[:8]
        print(f"[INVESTIGATION] #{short_id} CLOSED — Verdict: {verdict.upper()} after {duration}s. {verdict_reasoning}")
        
        self._save_investigation(inv)
        
        # Remove from active dict
        if investigation_id in self.active_investigations:
            del self.active_investigations[investigation_id]
            
        return inv

    def _load_active_investigations(self):
        try:
            if not self.memory_path.exists():
                return
            investigations = {}
            with open(self.memory_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        iid = data.get("investigation_id")
                        if iid:
                            investigations[iid] = data
                    except Exception:
                        continue
            
            # Repopulate active investigations
            for iid, inv in investigations.items():
                if inv.get("status") == "active":
                    self.active_investigations[iid] = inv
        except Exception as e:
            logger.warning("Error loading active investigations: %s", e)

    def _save_investigation(self, inv_data):
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(inv_data, ensure_ascii=True) + "\n")
        except Exception as e:
            logger.warning("Failed to save investigation record: %s", e)
