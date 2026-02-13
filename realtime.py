import numpy as np

buffer_audio = []

def audio_callback(frame):
    audio = frame.to_ndarray()

    # stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # dtype fix
    audio = audio.astype(np.float32)

    # normalize
    max_val = np.max(np.abs(audio)) + 1e-9
    audio = audio / max_val

    buffer_audio.append(audio)

    return frame
