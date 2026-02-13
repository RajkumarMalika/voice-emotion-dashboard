import streamlit as st
import tempfile, os
import plotly.express as px
import whisper
import numpy as np

from emotion_model import load_heavy_model, load_light_model
from utils import *
from realtime import audio_callback, buffer_audio
from streamlit_webrtc import webrtc_streamer

st.set_page_config(layout="wide")
st.title("🎙️ AI Voice Emotion Analytics")

dark = st.sidebar.toggle("🌙 Dark Mode")
mode = st.sidebar.radio("Mode", ["Upload Analysis", "Real-Time"])

@st.cache_resource
def get_heavy(): return load_heavy_model()

@st.cache_resource
def get_light(): return load_light_model()

@st.cache_resource
def get_whisper(): return whisper.load_model("tiny")

# -------- REALTIME --------
if mode == "Real-Time":
    st.subheader("🎤 Live Emotion")
    light_model = get_light()

    webrtc_streamer(key="mic", audio_frame_callback=audio_callback)

    if len(buffer_audio) > 12:
        chunk = np.concatenate(buffer_audio).astype(np.float32)

        energy = np.mean(np.abs(chunk))
        if energy < 0.02:
            st.warning("No strong voice detected")
        else:
            pred = light_model(chunk)[0]
            st.success(f"Emotion: {pred['label']}")

        buffer_audio.clear()

# -------- UPLOAD --------
else:
    heavy_model = get_heavy()
    whisper_model = get_whisper()

    file = st.file_uploader("Upload Audio", type=["wav","mp3"])
    if file:
        prog = st.progress(0)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        st.audio(file)
        prog.progress(20)

        st.plotly_chart(plot_waveform(path), width="stretch")
        prog.progress(40)

        result = whisper_model.transcribe(path)
        st.write(result["text"])
        prog.progress(60)

        chunks = split_audio(path)
        df = analyze_chunks(heavy_model, chunks)
        os.remove(path)
        prog.progress(80)

        st.dataframe(df)

        for _, r in df.iterrows():
            st.markdown(
                f"<span style='background:{r['Color']};padding:6px;border-radius:6px;color:white;'>"
                f"{r['Time']}s {r['Emotion']}</span>",
                unsafe_allow_html=True
            )

        st.subheader("Summary")
        for s in generate_summary(df): st.write(s)

        st.metric("Interview Score", f"{interview_score(df)}/10")
        st.plotly_chart(heatmap_chart(df), width="stretch")
        st.plotly_chart(confidence_gauge(df["Confidence"].mean()), width="stretch")

        st.plotly_chart(px.line(df, x="Time", y="Emotion"), width="stretch")
        st.plotly_chart(px.pie(df, names="Emotion"), width="stretch")

        prog.progress(100)
        st.download_button("Download CSV", df.to_csv(index=False), "report.csv")
