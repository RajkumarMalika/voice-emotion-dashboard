from transformers import pipeline

def load_heavy_model():
    return pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )

def load_light_model():
    return pipeline(
        "audio-classification",
        model="superb/hubert-base-superb-er"
    )
