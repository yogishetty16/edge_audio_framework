# Sample Data

Place `.wav` audio files in this folder for testing.

## How to Test

Run any audio file through the framework using:

```bash
python fast_run_file.py --file "sample_data/your_file.wav" --agent --tasks vad,asr,emotion
```

## Recommended Test Files

- A short speech clip (5-10 seconds) for testing ASR, VAD, emotion, and speaker ID
- A music clip for testing music genre classification
- An environmental sound clip (birds, traffic, alarms) for testing ESC
- A silence or white noise clip for testing anomaly detection

## Supported Format

- Format: WAV (mono or stereo)
- Sample rate: any (automatically resampled to 16kHz)
- Duration: 3 to 30 seconds recommended
