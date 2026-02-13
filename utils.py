import librosa
import pandas as pd
import plotly.graph_objects as go
import numpy as np

emotion_colors = {
    "happy": "green",
    "sad": "blue",
    "angry": "red",
    "neutral": "gray",
    "fear": "purple",
    "disgust": "brown",
    "surprise": "orange"
}

emotion_score_map = {
    "happy": 9,
    "surprise": 8,
    "neutral": 6,
    "sad": 4,
    "fear": 3,
    "angry": 2,
    "disgust": 2
}

# -------- SPLIT AUDIO --------
def split_audio(file_path=None, chunk_duration=3, y=None, sr=16000):
    if y is None:
        if file_path is None:
            raise ValueError("file_path or y must be provided")
        y, sr = librosa.load(file_path, sr=sr)
    chunk_samples = chunk_duration * sr
    chunks = []

    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i+chunk_samples]

        # silence removal
        if np.max(np.abs(chunk)) < 0.01:
            continue

        # normalize + dtype fix
        chunk = librosa.util.normalize(chunk)
        chunk = chunk.astype(np.float32)

        start_time = i / sr
        chunks.append((chunk, start_time))

    return chunks

# -------- ANALYZE --------
def analyze_chunks(model, chunks):
    results = []

    for chunk, start_time in chunks:
        pred = model(chunk)[0]
        emotion = pred['label'].lower()

        results.append({
            "Time": round(start_time, 2),
            "Emotion": emotion,
            "Confidence": round(pred['score'], 2),
            "Color": emotion_colors.get(emotion, "black"),
            "Score": emotion_score_map.get(emotion, 5)
        })

    return pd.DataFrame(results)

# -------- WAVEFORM --------
def plot_waveform(path=None, y=None, sr=16000):
    if y is None:
        if path is None:
            raise ValueError("path or y must be provided")
        y, sr = librosa.load(path, sr=sr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y))
    fig.update_layout(title="Audio Waveform")
    return fig

# -------- WOW FEATURES --------
def generate_summary(df):
    summary = []
    prev = None
    start = 0

    for _, row in df.iterrows():
        if row["Emotion"] != prev:
            if prev is not None:
                summary.append(f"{prev} from {start}s to {row['Time']}s")
            prev = row["Emotion"]
            start = row["Time"]

    summary.append(f"{prev} from {start}s to end")
    return summary


def interview_score(df):
    return round(df["Score"].mean(), 2)


def heatmap_chart(df):
    fig = go.Figure(
        data=go.Heatmap(
            z=[df["Score"].tolist()],
            x=df["Time"].tolist(),
            y=["Intensity"],
            colorscale="RdYlGn"
        )
    )
    return fig


def confidence_gauge(val):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        gauge={'axis': {'range': [0, 1]}}
    ))
    return fig
