"""
fast_run.py
===========
Speed-optimised version of run_live.py.

Key improvements over run_live.py:
  1. Pre-warms ALL models at startup (so first analysis is fast)
  2. Runs tasks IN PARALLEL with ThreadPoolExecutor
  3. Uses only fast tasks by default (skips slow 30s+ models)

Usage:
    python fast_run.py                     # fast 6 tasks, pre-warmed
    python fast_run.py --tasks all          # all 12 tasks in parallel
    python fast_run.py --duration 5         # 5 second recording
    python fast_run.py --device 1           # use mic device 1
    python fast_run.py --no-prewarm         # skip pre-warm (faster to start)
"""

# ── SSL fix FIRST ────────────────────────────────────────────────────────────
import ssl, os, sys
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"]           = "0"
os.environ["CURL_CA_BUNDLE"]              = ""
os.environ["REQUESTS_CA_BUNDLE"]          = ""
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # Fix Protobuf TypeError
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ["HF_HUB_VERBOSITY"]           = "error"

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"]    = os.path.join(_ROOT, "models", "hf_cache")
os.environ["TORCH_HOME"] = os.path.join(_ROOT, "models", "torch_hub")
sys.path.insert(0, _ROOT)

# ── Robust SSL Bypass for Corporate Proxy ───────────────────────────────────
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception: pass

try:
    import requests
    _orig_req = requests.Session.request
    def _patched_req(self, *args, **kwargs):
        kwargs["verify"] = False
        return _orig_req(self, *args, **kwargs)
    requests.Session.request = _patched_req
except Exception: pass

try:
    import httpx
    _orig_httpx = httpx.Client.__init__
    def _patched_httpx(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_httpx(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_httpx
except Exception: pass


# ── Imports ──────────────────────────────────────────────────────────────────
import time, argparse, warnings, json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

from agent import AudioAgent, AudioMemory
from agent.policy import PROFILE_POLICIES

# Fast tasks only by default (skip slow keyword_spotting & lang_accent_id)
FAST_TASKS = ["vad", "asr", "emotion", "speech_quality", "audio_quality_monitoring"]
QUICK_TASKS = ["vad", "asr", "audio_quality_monitoring", "anomaly_detection", "impulse_event"]
ALL_TASKS  = ["vad", "asr", "keyword_spotting", "speaker_id", "emotion",
              "speech_quality", "lang_accent_id", "esc", "acoustic_event_detection",
              "audio_quality_monitoring", "music_speech_detection", "anomaly_detection",
              "impulse_event"]

# Latency class for each task: FAST < 1s, MED 1-10s, SLOW > 10s
TASK_SPEED = {
    "vad": "FAST", "audio_quality_monitoring": "FAST", "speech_quality": "FAST",
    "anomaly_detection": "FAST", "impulse_event": "FAST", "asr": "MED", "speaker_id": "MED",
    "acoustic_event_detection": "MED", "emotion": "MED", "music_speech_detection": "MED",
    "esc": "SLOW", "keyword_spotting": "SLOW", "lang_accent_id": "SLOW",
}


def prewarm_models(tasks: list):
    """Load all models into RAM before recording starts."""
    from core.model_registry import registry
    import importlib

    print("\n[PREWARM] Loading models into memory...")
    t0 = time.perf_counter()

    # Map task names to their module
    task_modules = {
        "vad": "tasks.vad", "asr": "tasks.asr",
        "keyword_spotting": "tasks.keyword_spotting",
        "speaker_id": "tasks.speaker_id", "emotion": "tasks.emotion",
        "speech_quality": "tasks.speech_quality",
        "lang_accent_id": "tasks.accent_lang_id", "esc": "tasks.esc",
        "acoustic_event_detection": "tasks.acoustic_event_detection",
        "audio_quality_monitoring": "tasks.audio_quality_monitor",
        "music_speech_detection": "tasks.music_speech_detection",
        "anomaly_detection": "tasks.anomaly_detection",
        "impulse_event": "tasks.impulse_event",
    }

    def _load_one(task_name):
        try:
            mod = importlib.import_module(task_modules[task_name])
            # Trigger registry.get() to actually load the model weights
            return task_name, None
        except Exception as e:
            return task_name, str(e)

    # Load models in parallel threads
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_load_one, t): t for t in tasks if t in task_modules}
        for f in as_completed(futures):
            name, err = f.result()
            speed = TASK_SPEED.get(name, "?")
            if err:
                print(f"  [!] {name}: {err[:60]}")
            else:
                print(f"  [OK] {name} ({speed})")

    elapsed = time.perf_counter() - t0
    print(f"[PREWARM] Done in {elapsed:.1f}s — first analysis will be fast!\n")


def record_audio(duration_sec=5, device_index=None):
    import pyaudio
    pa = pyaudio.PyAudio()
    chunk = 1024
    stream = pa.open(rate=16000, channels=1, format=pyaudio.paFloat32,
                     input=True, frames_per_buffer=chunk,
                     input_device_index=device_index)
    print(f"\nRecording {duration_sec}s ... speak now!")
    for i in range(3, 0, -1):
        print(f"  {i}..."); time.sleep(1)
    print("  GO! Speak now.")
    frames = [stream.read(chunk, exception_on_overflow=False)
              for _ in range(int(16000 / chunk * duration_sec))]
    stream.stop_stream(); stream.close(); pa.terminate()
    data = np.frombuffer(b"".join(frames), dtype=np.float32)
    print(f"  Recorded {len(data)/16000:.1f}s\n")
    return data, 16000


def run_parallel(audio_np, sr, tasks):
    """Run all tasks in parallel using threads."""
    from core.audio_io import AudioData
    import importlib

    audio = AudioData(waveform=audio_np, sample_rate=sr,
                      duration_sec=len(audio_np)/sr, n_channels=1)

    task_modules = {
        "vad": "tasks.vad", "asr": "tasks.asr",
        "keyword_spotting": "tasks.keyword_spotting",
        "speaker_id": "tasks.speaker_id", "emotion": "tasks.emotion",
        "speech_quality": "tasks.speech_quality",
        "lang_accent_id": "tasks.accent_lang_id", "esc": "tasks.esc",
        "acoustic_event_detection": "tasks.acoustic_event_detection",
        "audio_quality_monitoring": "tasks.audio_quality_monitor",
        "music_speech_detection": "tasks.music_speech_detection",
        "anomaly_detection": "tasks.anomaly_detection",
        "impulse_event": "tasks.impulse_event",
    }

    def _run_task(task_name):
        try:
            mod = importlib.import_module(task_modules[task_name])
            t0 = time.perf_counter()
            result = mod.analyze(audio)
            ms = (time.perf_counter() - t0) * 1000
            r = result.__dict__ if hasattr(result, '__dict__') else {}
            r["latency_ms"] = ms
            return task_name, r
        except Exception as e:
            return task_name, {"success": False, "error": str(e)}

    print(f"Running {len(tasks)} tasks in parallel...")
    t0 = time.perf_counter()

    results = {}
    # Run FAST tasks first immediately, overlap with SLOW tasks
    with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as ex:
        futures = {ex.submit(_run_task, t): t for t in tasks if t in task_modules}
        for f in as_completed(futures):
            name, result = f.result()
            results[name] = result

    total_ms = (time.perf_counter() - t0) * 1000
    return results, total_ms


LABELS = {
    "vad": "Voice Activity Detection", "asr": "Automatic Speech Recognition",
    "keyword_spotting": "Keyword Spotting", "speaker_id": "Speaker ID",
    "emotion": "Emotion Detection", "speech_quality": "Speech Quality",
    "lang_accent_id": "Language / Accent ID", "esc": "Environmental Sound",
    "acoustic_event_detection": "Acoustic Events",
    "audio_quality_monitoring": "Audio Quality Monitor",
    "music_speech_detection": "Music / Speech Detection",
    "anomaly_detection": "Anomaly Detection",
    "impulse_event": "Impulse Event Detector",
}


def print_results(results, total_ms, tasks):
    print(f"\n{'='*55}")
    print(f"  RESULTS  (total wall time: {total_ms:.0f}ms = {total_ms/1000:.1f}s)")
    print(f"{'='*55}")

    for task_name in tasks:
        r = results.get(task_name)
        if not r:
            continue
        label = LABELS.get(task_name, task_name)
        if not r.get("success", True):
            print(f"\n  XX  {label}: {r.get('error','?')[:80]}")
            continue
        ms = f"[{r['latency_ms']:.0f}ms]" if r.get("latency_ms") else ""
        print(f"\n  OK  {label}  {ms}")

        if task_name == "vad":
            print(f"      Speech: {r.get('speech_ratio',0)*100:.1f}%  "
                  f"| Duration: {r.get('total_speech_sec',0):.2f}s")
        elif task_name == "asr":
            print(f"      >> \"{r.get('text','')[:120]}\"")
            print(f"      Language: {r.get('language','N/A')}")
        elif task_name == "keyword_spotting":
            print(f"      Keyword: {r.get('top_label','')} "
                  f"({r.get('top_score',0)*100:.1f}%)")
        elif task_name == "speaker_id":
            print(f"      Speakers: {r.get('num_speakers',0)}")
        elif task_name == "emotion":
            print(f"      Emotion: {r.get('top_emotion','')} "
                  f"({r.get('top_score',0)*100:.1f}%)")
            if r.get("sentiment"):
                print(f"      Sentiment: {r['sentiment']}")
        elif task_name == "speech_quality":
            print(f"      DNSMOS: {r.get('dnsmos_overall',0):.2f}/5  "
                  f"| SNR: {r.get('snr_db',0):.1f}dB")
        elif task_name == "lang_accent_id":
            print(f"      Language: {r.get('top_language','')}  "
                  f"| Accent: {r.get('top_accent','N/A')}")
        elif task_name == "esc":
            print(f"      Sound: {r.get('top_class','')} "
                  f"({r.get('top_score',0)*100:.1f}%)")
        elif task_name == "acoustic_event_detection":
            for ev in (r.get("events") or [])[:3]:
                # Acoustic event may be a dict or an AcousticEvent object
                lbl = getattr(ev, "label", ev.get("label", "")) if isinstance(ev, dict) else getattr(ev, "label", getattr(ev, "name", ""))
                scr = getattr(ev, "score", ev.get("score", 0)) if isinstance(ev, dict) else getattr(ev, "score", getattr(ev, "probability", 0))
                print(f"        - {lbl} ({scr*100:.1f}%)")
        elif task_name == "audio_quality_monitoring":
            print(f"      Quality: {r.get('quality_label','').upper()}  "
                  f"| MOS: {r.get('mos',0):.2f}  "
                  f"| SNR: {r.get('snr_db',0):.1f}dB  "
                  f"| Clipping: {r.get('clipping_detected',False)}")
        elif task_name == "music_speech_detection":
            print(f"      Speech: {r.get('speech_fraction',0)*100:.1f}%  "
                  f"| Music: {r.get('music_fraction',0)*100:.1f}%")
        elif task_name == "anomaly_detection":
            flag = "** ANOMALY **" if r.get("is_anomaly") else "Normal"
            print(f"      {flag}  | Score: {r.get('anomaly_score',0):.4f}")
        elif task_name == "impulse_event":
            flag = "** IMPULSE **" if r.get("is_impulse") else "Normal"
            print(f"      {flag}  | {r.get('event_label','')} "
                  f"({r.get('confidence',0)*100:.1f}%)")
            print(f"      Crest: {r.get('crest_factor',0):.1f}  "
                  f"| Transient: {r.get('transient_ratio',0):.1f}x  "
                  f"| High-band: {r.get('high_band_ratio',0)*100:.1f}%")

    print(f"\n{'='*55}\nDone!\n")


def print_agent_plan(title, plan):
    print(f"\n{'='*55}")
    print(f"  AGENT PLAN - {title}")
    print(f"{'='*55}")
    print(f"  Reason : {plan.reason}")
    print(f"  Tasks  : {', '.join(plan.tasks) if plan.tasks else 'none'}")
    if plan.skipped_tasks:
        print("  Skipped:")
        for task, reason in plan.skipped_tasks.items():
            print(f"    - {task}: {reason}")


def print_agent_decision(decision, show_json=False):
    payload = decision.to_dict()
    print(f"\n{'='*55}")
    print("  AGENT DECISION")
    print(f"{'='*55}")
    print(f"  What happened : {_humanize_event(payload['event_type'])}")
    print(f"  Priority      : {payload['priority'].upper()} ({payload['confidence'] * 100:.0f}% confidence)")
    print(f"  Next action   : {_humanize_action(payload['recommended_action'])}")
    print(f"  Privacy       : {_humanize_privacy(payload['privacy_mode'])}")

    reasons = payload["reasoning"][:3]
    if reasons:
        print("\n  Why:")
        for item in reasons:
            print(f"    - {item}")
    if payload["risk_signals"]:
        print(f"\n  Risk signals : {', '.join(_humanize_token(x) for x in payload['risk_signals'])}")
    if payload["follow_up_tasks"]:
        print(f"  Follow-up    : {', '.join(_humanize_token(x) for x in payload['follow_up_tasks'])}")

    failed = [name for name, status in payload["task_health"].items() if status != "ok"]
    if failed:
        print(f"  Task issues  : {', '.join(_humanize_token(x) for x in failed)}")

    print(f"  Models used  : {', '.join(_humanize_token(x) for x in payload['models_used'])}")

    if show_json:
        print("\n  Structured output:")
        print(json.dumps(payload, indent=2))


def print_memory_context(context, memory_path):
    if not context or context.get("recent_count", 0) == 0:
        return

    print(f"\n{'='*55}")
    print("  AUDIO MEMORY")
    print(f"{'='*55}")
    print(f"  Recent events : {context.get('recent_count', 0)} in last {context.get('window_sec', 0)}s")
    print(f"  Related       : {context.get('related_count', 0)}")
    if context.get("escalation_signals"):
        print("  Timeline cues :")
        for signal in context["escalation_signals"]:
            print(f"    - {signal}")

    related = context.get("related_events") or []
    if related:
        print("  Last related  :")
        for event in related[-3:]:
            summary = event.get("summary") or event.get("event_type", "")
            print(f"    - {summary}")
    print(f"  Store         : {memory_path}")


def finalize_agent_decision(agent, results, args, source="live"):
    decision = agent.decide(results)
    memory_context = None
    memory = None

    if not getattr(args, "no_agent_memory", False):
        memory = AudioMemory(getattr(args, "agent_memory_path", None))
        memory_context = memory.build_context(
            decision,
            window_sec=getattr(args, "agent_memory_window", 600),
        )
        memory.enrich_decision(decision, memory_context)

    print_agent_decision(decision, show_json=getattr(args, "agent_json", False))

    if memory is not None:
        memory.append_decision(decision, source=source)
        print_memory_context(memory_context, memory.path)


def _humanize_event(event_type):
    labels = {
        "possible_safety_incident": "Possible safety incident",
        "possible_impulse_threat": "Possible impulse threat",
        "acoustic_risk_event": "Acoustic risk event",
        "speech_workflow_event": "Speech/workflow event",
        "speech_detected": "Speech detected",
        "environmental_audio_event": "Environmental audio event",
        "normal_audio": "Normal audio",
    }
    return labels.get(event_type, _humanize_token(event_type))


def _humanize_action(action):
    labels = {
        "create_incident_and_notify_operator": "Create incident and notify operator",
        "store_metadata_and_request_review": "Store metadata and request review",
        "extract_intent_and_route_to_workflow": "Extract intent and route to workflow",
        "wait_for_more_context": "Wait for more context",
        "log_event": "Log event",
        "ignore_or_continue_monitoring": "Continue monitoring",
    }
    return labels.get(action, _humanize_token(action))


def _humanize_privacy(privacy_mode):
    labels = {
        "metadata_only": "Metadata only; raw audio stays local",
        "metadata_plus_redacted_evidence": "Metadata plus redacted evidence; raw audio stays local",
        "raw_audio_allowed": "Raw audio export allowed by policy",
    }
    return labels.get(privacy_mode, _humanize_token(privacy_mode))


def _humanize_token(value):
    return str(value).replace("_", " ").strip().capitalize()


def run_agent_orchestrated(audio_np, sr, agent, requested_tasks):
    initial_plan = agent.plan_initial_tasks(requested_tasks)
    print_agent_plan("INITIAL TRIAGE", initial_plan)

    results = {}
    total_ms = 0.0
    ordered_tasks = []

    if initial_plan.tasks:
        stage_results, stage_ms = run_parallel(audio_np, sr, initial_plan.tasks)
        results.update(stage_results)
        total_ms += stage_ms
        ordered_tasks.extend(initial_plan.tasks)

    follow_plan = agent.plan_follow_up_tasks(
        results,
        requested_tasks=requested_tasks,
        already_run=results.keys(),
    )
    print_agent_plan("FOLLOW-UP", follow_plan)

    if follow_plan.tasks:
        stage_results, stage_ms = run_parallel(audio_np, sr, follow_plan.tasks)
        results.update(stage_results)
        total_ms += stage_ms
        ordered_tasks.extend(follow_plan.tasks)

    return results, total_ms, ordered_tasks


def main():
    parser = argparse.ArgumentParser(description="Fast parallel audio analysis")
    parser.add_argument("--duration", "-d", type=int, default=5)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--tasks", "-t", default=None)
    parser.add_argument("--no-prewarm", action="store_true",
                        help="Skip pre-warming (quicker to start, slower first run)")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--agent", action="store_true",
                        help="Enable the agentic edge audio decision layer")
    parser.add_argument("--agent-mode", choices=["advisory", "orchestrated"],
                        default="advisory",
                        help="advisory runs selected tasks then decides; orchestrated stages tasks dynamically")
    parser.add_argument("--agent-profile", choices=sorted(PROFILE_POLICIES.keys()),
                        default="balanced",
                        help="Policy profile for task planning, risk, and privacy decisions")
    parser.add_argument("--agent-json", action="store_true",
                        help="Print the full structured agent JSON output")
    parser.add_argument("--no-agent-memory", action="store_true",
                        help="Disable persistent agent timeline memory for this run")
    parser.add_argument("--agent-memory-path", default="agent_memory/events.jsonl",
                        help="Path for persistent agent timeline memory")
    parser.add_argument("--agent-memory-window", type=int, default=600,
                        help="Seconds of recent memory used for timeline context")
    parser.add_argument("--clear-agent-memory", action="store_true",
                        help="Clear persistent agent timeline memory and exit")
    parser.add_argument("--quick", action="store_true",
                        help="Fast startup mode: skip prewarm and run only essential lightweight tasks")
    args = parser.parse_args()

    if args.clear_agent_memory:
        memory = AudioMemory(args.agent_memory_path)
        memory.clear()
        print(f"Cleared agent memory: {memory.path}")
        return

    if args.list_devices:
        import pyaudio; pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']}")
        pa.terminate(); return

    requested_tasks = None
    if args.tasks:
        requested_tasks = (ALL_TASKS if args.tasks.lower() == "all"
                           else [t.strip() for t in args.tasks.split(",") if t.strip()])

    if args.quick:
        requested_tasks = requested_tasks or QUICK_TASKS
        args.no_prewarm = True
        if args.agent:
            args.agent_mode = "advisory"

    if args.agent and args.agent_mode == "orchestrated":
        task_list = requested_tasks or ALL_TASKS
    else:
        task_list = requested_tasks or FAST_TASKS

    agent = AudioAgent(args.agent_profile) if args.agent else None

    print("="*55)
    print("  Edge Audio Framework — FAST MODE")
    print("  [Tasks run in PARALLEL]")
    print("="*55)
    print(f"  Duration : {args.duration}s")
    print(f"  Tasks    : {', '.join(task_list)}")
    if agent:
        print(f"  Agent    : {args.agent_mode} / {args.agent_profile}")
    if args.quick:
        print("  Quick    : enabled, slow model prewarm skipped")

    # Pre-warm models BEFORE recording
    if not args.no_prewarm:
        if agent and args.agent_mode == "orchestrated":
            prewarm_models(agent.plan_initial_tasks(task_list).tasks)
        else:
            prewarm_models(task_list)

    # Record audio
    audio_np, sr = record_audio(duration_sec=args.duration,
                                device_index=args.device)

    if agent and args.agent_mode == "orchestrated":
        results, total_ms, ordered_tasks = run_agent_orchestrated(audio_np, sr, agent, task_list)
        task_list = ordered_tasks
    else:
        # Run all tasks in parallel
        results, total_ms = run_parallel(audio_np, sr, task_list)

    print_results(results, total_ms, task_list)
    if agent:
        finalize_agent_decision(agent, results, args, source="live")


if __name__ == "__main__":
    main()
