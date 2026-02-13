from transformers import pipeline

def _load_pipeline(model_name):
    try:
        return pipeline(
            "audio-classification",
            model=model_name,
            model_kwargs={"use_safetensors": True}
        )
    except ValueError as e:
        msg = str(e)
        if "torch.load" in msg or "safetensors" in msg:
            raise RuntimeError(
                "Model load failed due to torch.load safety restrictions. "
                "Upgrade torch to >= 2.6 or use a model that provides safetensors."
            ) from e
        raise

def load_heavy_model():
    return _load_pipeline(
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )

def load_light_model():
    return _load_pipeline(
        "superb/hubert-base-superb-er"
    )
