import streamlit as st
import numpy as np
import cv2
import librosa
import whisper
import soundfile as sf
from tensorflow.keras.models import load_model
from audiorecorder import audiorecorder
import os
import uuid

st.set_page_config(layout="centered")

# SESSION STATE
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

# Lazy load Whisper only when needed
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None

# LOAD KERAS MODEL (cached)
@st.cache_resource(show_spinner="Loading AI model...")
def load_keras_model():
    return load_model("model_v2.keras")

model = load_keras_model()

# LANGUAGE MAP
whisper_lang_map = {
    "en": "english",
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "ml": "malayalam",
    "kn": "kannada",
    "mr": "marathi",
    "bn": "bengali"
}


# UI
st.title("🎤 AI vs Human Voice Detector")

uploaded_file = st.file_uploader("📁 Upload Audio", type=["wav", "mp3"])

st.markdown("### 🎙️ Record Voice")
audio_rec = audiorecorder("Click to record", "Recording...")

col1, col2 = st.columns(2)


# DELETE BUTTON 
with col1:
    if st.button("🗑️ Delete / Re-record"):
        st.session_state.audio_file = None
        st.rerun()

# STORE AUDIO
if uploaded_file is not None:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File too large (max 10MB)")
        st.stop()
    st.session_state.audio_file = uploaded_file

elif len(audio_rec) > 0:
    st.session_state.audio_file = audio_rec

if st.session_state.audio_file is None:
    st.stop()


# SAVE AUDIO 
temp_input = f"input_{uuid.uuid4().hex}"
file_name = f"temp_{uuid.uuid4().hex}.wav"

try:
    if hasattr(st.session_state.audio_file, "read"):
        # Save original upload (mp3/wav)
        with open(temp_input, "wb") as f:
            f.write(st.session_state.audio_file.read())

        # Convert safely using librosa
        audio, sr = librosa.load(temp_input, sr=16000)
        sf.write(file_name, audio, 16000)

        os.remove(temp_input)

    else:
        # Recorded audio
        st.session_state.audio_file.export(file_name, format="wav")

except Exception as e:
    st.error(f"Audio processing error: {e}")
    st.stop()

st.audio(file_name)

# AUDIO PROCESSING 
audio, sr = librosa.load(file_name, sr=None)

# ONLY resample (same as training)
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# WHISPER (lazy load)
if st.session_state.whisper_model is None:
    with st.spinner("Loading Whisper model..."):
        st.session_state.whisper_model = whisper.load_model("tiny")

try:
    result = st.session_state.whisper_model.transcribe(file_name, fp16=False)
    raw_lang = result["language"]
    text = result["text"]
    language = whisper_lang_map.get(raw_lang, "unknown")
except:
    language = "unknown"
    text = "Transcription failed"

if language == "unknown":
    st.error("Unsupported language. Please record again.")
    st.stop()

# CNN (AI vs HUMAN)
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=16000,
    n_mels=128,
    fmax=8000
)

log_mel = librosa.power_to_db(mel_spec)

img = cv2.resize(log_mel, (128, 128))
img = (img - img.min()) / (img.max() - img.min())
img = np.stack([img] * 3, axis=-1)
img = np.expand_dims(img, axis=0)

# Predict
type_pred, _ = model.predict(img, verbose=0)

prob = float(type_pred[0][0])

label = "AI" if prob > 0.5 else "Human"
confidence = prob if label == "AI" else (1 - prob)

# LANGUAGE CONFIDENCE
lang_conf = min(0.95, max(0.6, len(text) / 60))

# OUTPUT
st.markdown("## 🔍 Prediction Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("Voice Type", label)
    st.metric("Type Confidence", f"{confidence:.2f}")
    st.progress(confidence)

with col2:
    st.metric("Language", language)
    st.metric("Language Confidence", f"{lang_conf:.2f}")
    st.progress(lang_conf)

# Alerts
if label == "AI":
    st.warning("⚠️ This voice may be AI-generated")
else:
    st.success("✅ This appears to be a human voice")

# Transcription
st.markdown("### 📝 Transcription")
st.write(text)

if os.path.exists(file_name):
    os.remove(file_name)