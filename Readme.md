# 🎤 Multilingual AI vs Human Voice Detection System

A deep learning-based web application that detects whether a voice is AI-generated or human, and identifies the spoken language using speech recognition.

Built using:

* CNN (Convolutional Neural Network) for voice classification
* Mel Spectrograms for audio feature extraction
* OpenAI Whisper for transcription and language detection
* Streamlit for interactive UI


# 🚀 Features

* 🎧 Upload or record audio (WAV / MP3)
* 🤖 Detect AI vs Human voice
* 🌍 Identify language (8 supported languages)
* 📝 Generate speech transcription
* 📊 Display confidence scores
* 🔄 Re-record / delete audio functionality
* ⚡ Fast and optimized for real-time inference


# 🧠 Project Architecture

Audio Input
   ↓
Preprocessing (Resampling → 16kHz)
   ↓
Mel Spectrogram Generation
   ↓
Image Conversion (128×128×3)
   ↓
CNN Model
   ├── Voice Type (AI / Human)
   └── Language (Optional)
   ↓
Whisper Model
   ├── Language Detection
   └── Transcription
   ↓
Streamlit UI Output

# 📊 Model Details

# CNN Architecture

* Input: `(128, 128, 3)` spectrogram image
* 3 Convolution Blocks:
  * Conv2D (32) + MaxPooling
  * Conv2D (64) + MaxPooling
  * Conv2D (128) + MaxPooling
* Flatten Layer
* Dense (128) + Dropout
* Dual Outputs:

  * Type Output → AI vs Human (Sigmoid)
  * Language Output → 8 classes (Softmax)


# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/ai-vs-human-voice.git
cd ai-vs-human-voice
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv myenv
```

## Activate (Windows)

```bash
myenv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Required Versions 

👉 These versions are required to avoid compatibility errors.

```
streamlit==1.32.0
numpy==1.26.4
opencv-python==4.9.0.80
librosa==0.10.1
soundfile==0.12.1
tensorflow==2.13.0
keras==2.13.1
torch==2.1.2
openai-whisper==20231117
audiorecorder==0.1
ffmpeg-python==0.2.0
```

---

# ⚠️ IMPORTANT: Install FFmpeg

Whisper requires FFmpeg.

### Windows Setup:

1. Download from:
   👉 https://ffmpeg.org/download.html

2. Extract and add **bin folder** to PATH

Example:

```
C:\ffmpeg\bin
```

3. Verify:

```bash
ffmpeg -version
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

# 📁 Project Structure

```
📦 ai-vs-human-voice
 ┣ 📜 app.py
 ┣ 📜 model_v2.keras
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┗ 📂 dataset
```

---

# 🎯 How It Works

# Step 1: Audio Input
* User uploads or records audio

# Step 2: Preprocessing
* Resampled to 16kHz
* Normalized and trimmed
* Fixed length for model consistency

# Step 3: Feature Extraction
* Convert audio → Mel Spectrogram
* Convert spectrogram → image

# Step 4: CNN Prediction
* Detects AI vs Human voice

# Step 5: Whisper Processing
* Detects *anguage
* Generates transcription


# ❓ Why 16kHz Sampling?

* Human speech lies within **0–8000 Hz**
* Nyquist theorem → minimum sampling = **16kHz**
* Reduces computation while preserving speech quality

---

# 🧪 Limitations

* Accuracy depends on audio quality
* Whisper may misinterpret noisy input
* Real-time inference may be slow on CPU

---

#  Future Improvements

* Real-time streaming detection
* Improved AI voice dataset
* Transformer-based audio models
* Deployment on cloud (AWS / GCP)
* Mobile app integration

# 👨‍💻 Author
Lisha Malu

# 📜 License
This project is for academic and educational purposes.


# ⭐ Acknowledgements
* OpenAI Whisper
* TensorFlow / Keras
* Librosa Audio Processing
* Streamlit UI Framework


