"""
test_qai_whisper.py
===================
Standalone test script to verify Qualcomm AI Hub ONNX models.
"""

import sys
import os

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is not installed. Run: pip install onnxruntime")
    sys.exit(1)

# Path to the extracted folder containing encoder.onnx and decoder.onnx
# The user extracted it here:
MODEL_DIR = r"F:\edge_audio_framework\models\whisper_base-precompiled_qnn_onnx-float-qualcomm_snapdragon_x2_elite\whisper_base-precompiled_qnn_onnx-float-qualcomm_snapdragon_x2_elite"

def test_whisper_onnx():
    print("=" * 60)
    print("  Qualcomm AI Hub Whisper ONNX Test (QNN Context Binary)")
    print("=" * 60)

    encoder_path = os.path.join(MODEL_DIR, "encoder.onnx")
    decoder_path = os.path.join(MODEL_DIR, "decoder.onnx")

    if not os.path.exists(encoder_path):
        print(f"\n[FAIL] Encoder not found at: {encoder_path}")
        return

    print(f"Testing ONNX Runtime capabilities...")
    print(f"Available Execution Providers: {ort.get_available_providers()}")
    
    print(f"\nAttempting to load QNN-compiled Encoder: {encoder_path}")
    try:
        # These models usually require the QNNExecutionProvider because they are pre-compiled for Snapdragon.
        # We will try loading them with default providers first.
        session = ort.InferenceSession(encoder_path)
        print("  [OK] Model loaded successfully into memory!")
    except Exception as e:
        print(f"\n  [FAIL] Could not load ONNX model.")
        print(f"  Error details:\n{e}\n")
        print("-" * 60)
        print("DIAGNOSIS:")
        print("This error usually means your Linux machine does NOT have a Qualcomm Snapdragon processor.")
        print("The files you downloaded from the Qualcomm AI Hub contain 'qairt_context.bin', which are machine-code binaries locked to Snapdragon Neural Processing Units (NPUs).")
        print("If your office machine is a standard Intel or AMD computer, it physically cannot run these Snapdragon-specific files.")

if __name__ == "__main__":
    test_whisper_onnx()
