# 🎵 Audio Spectrogram Viewer

A **PyQt5-based desktop application** for visualizing audio files as **FFT spectrograms** in real time.  
Supports **playback, zooming, saving spectrograms, and live playback cursor** synchronized with audio.  

---

## ✨ Features
- 📂 Load audio files (`.mp3`, `.wav`, `.flac`)
- 🎚️ Play / Pause / Stop playback
- ⏩ Seek with a slider
- 🔍 Zoom in & out on spectrogram
- 📊 Real-time FFT spectrogram generation
- 🖼 Save spectrogram as PNG
- 🎧 Playback with **pygame mixer**
- ⚡ Smooth scrolling with playback cursor
- Logging for debugging

---

## 📦 Requirements

- Python 3.9+
- Install dependencies:
```bash
pip install numpy matplotlib PyQt5 soundfile pygame
🚀 Run the app
bash
Copy code
python main.py
🖼 UI Preview
Load & Play	Zoomed Spectrogram	Save Spectrogram

🛠 Directory Structure
css
Copy code
audio-spectrogram-viewer/
├── main.py              # Main application
├── images/              # Screenshots
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
└── README.md
🎹 Usage
Launch the app: python main.py

Click "Загрузить аудиофайл" to select an audio file

Use:

▶️ Play/Pause

⏹ Stop

⏩ Slider for seeking

🔍 Zoom In / Out buttons

Save spectrogram → "Сохранить спектрограмму"

⚠️ Notes
FFmpeg may be required for reading .mp3 and .flac files.

Spectrogram rendering may be CPU intensive for long audio files.

The app currently supports single-channel (mono) audio — stereo is automatically reduced to one channel.
