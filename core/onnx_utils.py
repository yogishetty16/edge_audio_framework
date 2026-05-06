"""
core/onnx_utils.py
==================
ONNX Runtime session factory with sensible CPU defaults.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_onnx_session(model_path: str, num_threads: int = 2):
    """
    Create an ONNXRuntime InferenceSession optimised for CPU edge inference.

    Parameters
    ----------
    model_path  : Path to the .onnx model file.
    num_threads : Number of intra/inter-op threads (default 2 for edge devices).
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = num_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3  # suppress verbose output

    providers = ["CPUExecutionProvider"]
    try:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    logger.info(f"[onnx_utils] Loaded: {model_path} | providers={session.get_providers()}")
    return session
