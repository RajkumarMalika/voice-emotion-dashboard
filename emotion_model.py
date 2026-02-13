from transformers import pipeline

def load_heavy_model():
    return pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )

def load_light_model():
    try:
        # Try to load the light model
        return pipeline(
            "audio-classification",
            model="superb/hubert-base-superb-er"
        )
    except Exception as e:
        # If the light model fails, try an alternative
        print(f"Failed to load superb/hubert-base-superb-er: {e}")
        print("Trying alternative model...")
        return pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
