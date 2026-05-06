"""
tasks/music_genre.py
======================
Music Genre Classification task for the Edge Audio Framework (Use Case 1).
Uses a lightweight DistilHuBERT model fine-tuned on the GTZAN dataset.

Model: sanchit-gandhi/distilhubert-finetuned-gtzan
Outputs 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
"""

import logging
import torch
from dataclasses import dataclass
from typing import Dict, Any

from core.audio_io import AudioData
from core.model_registry import registry, DEVICE, MODELS_DIR

logger = logging.getLogger(__name__)

MUSIC_GENRE_MODEL_ID = "sanchit-gandhi/distilhubert-finetuned-gtzan"


@dataclass
class MusicGenreResult:
    top_genre: str
    top_score: float
    all_scores: Dict[str, float]


def _load_music_genre_model() -> Dict[str, Any]:
    """Lazy loader for the DistilHuBERT GTZAN model."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    
    logger.info(f"Loading Music Genre Model (GTZAN): {MUSIC_GENRE_MODEL_ID}")
    
    try:
        extractor = AutoFeatureExtractor.from_pretrained(
            MUSIC_GENRE_MODEL_ID,
            cache_dir=str(MODELS_DIR / "hf_cache")
        )
        model = AutoModelForAudioClassification.from_pretrained(
            MUSIC_GENRE_MODEL_ID,
            cache_dir=str(MODELS_DIR / "hf_cache")
        )
        model.to(DEVICE)
        model.eval()
        logger.info("Music Genre model loaded successfully.")
        return {"extractor": extractor, "model": model}
    except Exception as e:
        logger.error(f"Failed to load Music Genre model: {e}")
        return None

# Register the loader
registry.register("music_genre", _load_music_genre_model)


def analyze(audio: AudioData) -> MusicGenreResult:
    """
    Classify the music genre in the provided audio.
    """
    bundle = registry.get("music_genre")
    if bundle is None:
        raise RuntimeError("Music Genre model failed to load.")
        
    extractor = bundle["extractor"]
    model = bundle["model"]
    
    # Needs to be 16kHz
    waveform = audio.waveform
    if audio.sample_rate != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=audio.sample_rate, target_sr=16000)

    # Process through the Hugging Face extractor
    inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
    # Get top prediction
    predicted_idx = torch.argmax(probs).item()
    top_genre = model.config.id2label[predicted_idx]
    top_score = probs[predicted_idx].item()
    
    # Store all scores
    all_scores = {
        model.config.id2label[i]: float(probs[i])
        for i in range(len(probs))
    }

    return MusicGenreResult(
        top_genre=top_genre,
        top_score=top_score,
        all_scores=all_scores
    )

