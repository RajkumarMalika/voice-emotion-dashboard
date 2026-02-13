# voice-emotion-dashboard
# 🎙️ AI Voice Emotion Analytics Dashboard

An AI-powered web application that analyzes voice audio and detects emotions in real-time and from uploaded files.
The system visualizes emotions on an interactive dashboard with waveform graphs, heatmaps, summaries, and interview scoring.

## 🚀 Features
## 🎤 Real-Time Emotion Detection

## 📂 Upload Audio Analysis

Supports WAV / MP3

Multi-language speech recognition

Emotion timeline with timestamps

CSV report download

## 📊 Dashboard Visualizations

Audio Waveform Graph

Emotion Timeline Line Chart

Emotion Distribution Pie Chart

Emotion Heatmap (WOW Feature)

Confidence Gauge Meter

Auto Emotion Summary
Example: “Speaker was angry from 10s–22s”

## 🎨 UI Enhancements

Dark / Light Mode

Emotion Color Badges

Progress Bar During Analysis

Clean Headings & Layout

## 🧠 AI Models Used
Task	Model
Speech-to-Text	OpenAI Whisper (tiny/base)
Emotion Detection (Upload)	Wav2Vec2 Large Emotion Recognition
Emotion Detection (Real-Time)	Lightweight Audio Classification Model
### 🛠️ Tech Stack

## Python

Streamlit – UI Framework

Librosa – Audio Processing

Transformers (HuggingFace) – Emotion Models

Whisper – Transcription

Plotly – Interactive Charts

NumPy / Pandas – Data Handling

streamlit-webrtc – Real-Time Audio

## 📁 Project Structure
voice-emotion-dashboard/
│
├── app.py                 # Main Streamlit App
├── utils.py               # Audio processing & charts
├── emotion_model.py       # Model loaders
├── realtime.py            # Microphone buffer logic
├── requirements.txt
└── README.md

## ⚙️ Installation
### 1. Clone Repository
git clone <your-repo-link>
cd voice-emotion-dashboard

### 2. Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### ▶️ Run Application
streamlit run app.py


### Open browser at:

http://localhost:8501

## 🧩 Future Improvements

Speaker Identification

Emotion Comparison Between Speakers

Mobile Optimization

GPU Acceleration

Fine-Tuned Custom Dataset

## 👨‍💻 Author

### Rajkumar Malik
