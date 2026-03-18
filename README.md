---
title: Interview_Coach
app_file: app.py
sdk: gradio
sdk_version: 5.20.0
---
# 🧔 AI Interview Coach: Next-Gen Career Preparation

Elevate your career preparation with a sophisticated, AI-driven interview simulation. The **AI Interview Coach** leverages high-performance Large Language Models (LLMs) and local Speech-to-Text inference to provide a hyper-realistic, low-latency, and insightful practice environment.

---

## 🚀 Key Innovation: The Smart Gatekeeper
Unlike generic AI tools, our coach includes a **Technical Resume & JD Validator**. 
- **Automatic Validation**: Before the session begins, the system verifies if the uploaded PDF is a legitimate Resume and if the description provided is a valid Job Requirement.
- **Access Control**: Nonsensical documents or random text inputs are immediately flagged, preventing credit waste and ensuring a professional simulation.

## ✨ Advanced Features

### 🎙️ Immersive Voice Interaction
- **Real-time Transcription**: Powered by `Faster-Whisper` (Standard tiny model) for millisecond-latency speech recognition.
- **Autonomous Conversationalist**: The AI doesn't just ask questions; it listens and follows up based on your previous answers using LLaMA 3.3 70B.
- **HR Conclusion**: At the end of the session, receive a spoken summary of your performance in a neutral, professional HR tone.

### 📊 Professional-Grade Analytics
- **Color-Coded Feedback**:
  - <span style="color: #92fe9d; font-weight: bold;">[STRENGTHS]</span>: Highlighting where you excelled.
  - <span style="color: #ff4b4b; font-weight: bold;">[WEAKNESSES]</span>: Pinpointing specific gaps in knowledge or delivery.
  - <span style="color: #ffcc00; font-weight: bold;">[NOT READY YET]</span>: Topics where you need more hands-on experience.
- **Interactive Radar Maps**: Visualize your competency across Communication, Technical Depth, Problem Solving, Confidence, and Cultural Fit.
- **Industry Benchmarking**: Compare your calculated scores against live peer benchmarks for your specific role.

### 🌘 Premium Design System
- **Glassmorphic UI**: A custom-built dark interface with cyan and green accents.
- **Responsive Layout**: Seamlessly transitions from high-resolution desktop monitors to mobile devices.
- **Interactive HR Character**: A fixed on-screen coach that tracks mouse movements and provides helpful tips via a FAQ chat bubble.

---

## 🛠️ Architecture & Tech Stack

```mermaid
graph TD
    A[User Uploads Resume & JD] --> B{Gatekeeper Validation}
    B -- Invalid --> C[Access Denied Error]
    B -- Valid --> D[Resume/JD Summary Generation]
    D --> E[Interactive Interview Loop]
    E --> F[ASR: Faster-Whisper]
    F --> G[LLM: Groq LLaMA 3.3]
    G --> H[TTS: gTTS]
    H --> E
    E --> I[Final Evaluation]
    I --> J[Performance Charts & Spoken Conclusion]
```

- **Frontend**: Gradio (Custom CSS/JS injected for premium dark theme).
- **Core Intelligence**: Groq API (LLaMA-3.3-70B-Versatile for extreme speed).
- **Audio Processing**: Faster-Whisper (In-IDE local inference).
- **PDF Extraction**: PyPDF2.
- **Data Viz**: Plotly.

---

## ⚙️ Detailed Setup Guide

### 1. Prerequisites
- **Python**: 3.10 or 3.11 recommended.
- **FFmpeg**: Required for audio processing. Install via `brew install ffmpeg` (Mac) or `choco install ffmpeg` (Windows).

### 2. Installation
```bash
# Clone the repo
git clone https://github.com/coder-apr-5/Interview_Coach.git
cd Interview_Coach

# Create environment
python -m venv venv
./venv/Scripts/activate  # Windows

# Install wheels first for audio (optional but recommended)
pip install setuptools wheel

# Install core requirements
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file:
```env
GROQ_API_KEY=gsk_your_key_here
```

### 4. Running the App
```bash
python myapp.py
```
Wait for `Pre-loading Whisper model...` and `Whisper ready.` before starting your first session for the best experience.

---

## ✨ Detailed Features Breakdown

### 1. The Smart Gatekeeper AI
The system uses a two-stage validation process before starting any interview:
- **Resume Verification**: Uses LLM logic to differentiate between a professional resume and random text.
- **JD Depth Check**: Ensures the job description has enough context (skills, role, company) to generate meaningful questions.

### 2. Dynamic Follow-up Logic
Unlike static interview bots, our LLM remembers your previous answers. If you mention a specific project in Question 1, Question 2 might be a deep-dive into that project, simulating a real conversational interview flow.

### 3. Neutral HR Concluding Feedback
Upon completion, the AI generates a spoken conclusion. This message is designed to be **emotionally neutral** but behaviorally accurate—providing direct feedback on whether you met the professional standard for the role.

---

## 🧠 How the AI Thinks (Pipeline)

1. **Input Stage**: Python extracts raw text from PDF and user inputs.
2. **Analysis Stage**: LLaMA 3.3 creates a "Context Map" of your skills vs. job requirements.
3. **Session Stage**: 
    - **Whisper (ASR)**: Converts your spoken audio to text locally.
    - **Groq (LLM)**: Analyzes the transcription and generates the next follow-up.
    - **gTTS (TTS)**: Converts the AI's question into high-quality speech.
4. **Evaluation Stage**: A secondary "Senior HR" persona reviews the entire chat history to generate unbiased scores and a correction roadmap.

---

## 🛠️ Troubleshooting & FAQs

**Q: The microphone isn't working/recording.**
- A: Ensure you are using a browser that supports `navigator.mediaDevices` (Chrome/Edge/Firefox) and that you have granted permission to the site.

**Q: The AI is taking too long to reply.**
- A: Check your `GROQ_API_KEY`. The Groq API is usually ultra-fast (sub-second), but rate limits or an invalid key can cause delays.

**Q: Whisper model loading is slow.**
- A: The first time you run the app, FFmpeg and the Whisper weights (tiny model) are downloaded. This only happens once.

---

## 📈 Performance Visuals
The app generates high-fidelity **Plotly** charts:
- **Radar Charts**: Perfect for showing balanced skillsets.
- **Bar Charts**: Benchmarked against internal datasets for realistic comparison.

---

## 👨‍💻 Developed By

**Apurba Roy**
*Driven by the goal of making career coaching accessible to everyone.*

- [LinkedIn](https://linkedin.com/in/apurba-roy05)
- [GitHub](https://github.com/coder-apr-5)
- [Mail](mailto:apurbaroy.leo5@gmail.com)

---
© 2026 AI Interview Coach • **Master Your Next Big Role.**
