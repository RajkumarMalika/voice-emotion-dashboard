import streamlit as st
import tempfile, os
import plotly.express as px
import whisper
import numpy as np
import librosa
import soundfile as sf

from emotion_model import load_heavy_model, load_light_model
from utils import *
from realtime import audio_callback, buffer_audio
from streamlit_webrtc import webrtc_streamer

st.set_page_config(layout="wide")
st.title("🎙️ AI Voice Emotion Analytics")

# Initialize session state
if 'processed_file_id' not in st.session_state:
    st.session_state.processed_file_id = None

dark = st.sidebar.toggle("🌙 Dark Mode")
mode = st.sidebar.radio("Mode", ["Upload Analysis", "Real-Time"])

@st.cache_resource
def get_heavy(): return load_heavy_model()

@st.cache_resource
def get_light(): return load_light_model()

@st.cache_resource
def get_whisper():
    return whisper.load_model("tiny", device="cpu")

# -------- REALTIME --------
if mode == "Real-Time":
    st.subheader("🎤 Live Emotion")
    
    try:
        light_model = get_light()
    except Exception as e:
        st.error(f"❌ Failed to load real-time emotion model: {str(e)}")
        st.info("This might be due to model download issues or configuration problems.")
        st.stop()

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
    use_heavy_model = st.sidebar.checkbox("Use heavy model (more memory)", value=False)
    enable_transcription = st.sidebar.checkbox("Transcribe with Whisper", value=True)

    file = st.file_uploader("Upload Audio", type=["wav","mp3"])
    if file:
        # Generate unique file ID
        file_id = f"{file.name}_{file.size}"
        
        # Check if this is a new file
        if st.session_state.processed_file_id != file_id:
            st.session_state.processed_file_id = file_id
            
        prog = st.progress(0)
        path = None
        
        try:
            # Create temp file from uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file.getvalue())
                tmp.flush()
                path = tmp.name

            st.audio(file)
            prog.progress(20)

            # Load audio once to avoid repeated disk reads
            audio_data, sr = librosa.load(path, sr=16000)
            st.plotly_chart(plot_waveform(path=path, sr=sr), width="stretch")
            prog.progress(40)

            # Check audio energy
            energy = np.mean(np.abs(audio_data))
            if energy < 0.02:
                st.warning("⚠️ Silence detected - please upload audio with voice")
                st.stop()

            if enable_transcription:
                whisper_model = get_whisper()
                result = whisper_model.transcribe(path)
                st.write("Transcript:", result["text"])
            prog.progress(60)

            model = get_heavy() if use_heavy_model else get_light()
            chunks = split_audio(y=audio_data, sr=sr)
            df = analyze_chunks(model, chunks)
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
            
        except FileNotFoundError as e:
            st.error(f"❌ Audio file not found: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            # Always cleanup temp file
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
    else:
        # Reset when no file uploaded
        st.session_state.processed_file_id = None
