# 🧔 AI Interview Coach

Elevate your career preparation with a next-gen, AI-powered interview simulation experience. The **AI Interview Coach** uses state-of-the-art Large Language Models (LLMs) and Speech-to-Text technology to provide real-time, tailored interview practice and deep performance analytics.

![Premium UI Mockup](https://raw.githubusercontent.com/coder-apr-5/Interview_Coach/main/logo.png) (Replace with your actual repo image path)

## ✨ Features

- **🚀 Smart Initialization**: Analyzes your Resume (PDF) and the target Job Description to generate highly specific technical and behavioral questions.
- **🎙️ Realistic Voice Interaction**: Features an interactive AI Interviewer that speaks to you. Respond using your microphone for a true-to-life experience.
- **🛡️ Access Control & Validation**: Built-in "Gatekeeper" AI that detects and rejects invalid PDFs or nonsensical job descriptions to maintain quality.
- **📊 Detailed HR Evaluation**:
  - **Color-Coded Feedback**: Instant visual cues for <span style="color: #92fe9d">Strengths</span>, <span style="color: #ff4b4b">Weaknesses</span>, and <span style="color: #ffcc00">Areas of Correction</span>.
  - **Performance Radar**: A 360-degree view of your core competencies (Communication, Technical, etc.).
  - **Peer Benchmarking**: Compare your results against industry standard averages.
- **🌑 Premium Dark Interface**: A sleek, glassmorphic design optimized for both desktop and mobile devices.
- **💬 Direct Feedback Integration**: Real-time feedback form integrated with Formspree for seamless bug reporting or suggestions.

## 🛠️ Tech Stack

- **Framework**: [Gradio](https://gradio.app/) for the interactive web interface.
- **Intelligence**: [Groq](https://groq.com/) API (LLaMA 3.3 70B) for ultra-fast response times.
- **Voice Recognition**: [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (AI-powered Speech-to-Text).
- **Speech Synthesis**: [gTTS](https://github.com/pndurette/gTTS) (Google Text-to-Speech).
- **Visualization**: [Plotly](https://plotly.com/python/) for interactive charts.

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/keys)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/coder-apr-5/Interview_Coach.git
   cd Interview_Coach
   ```

2. **Setup Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_actual_key_here
   ```

5. **Run the Application**
   ```bash
   python myapp.py
   ```

## 👥 Usage

1. **Upload Resume**: Drop your PDF resume into the upload box.
2. **Paste JD**: Copy-paste the job requirements you are preparing for.
3. **Start Interview**: The AI will greet you and ask the first question.
4. **Speak**: Use the microphone component to record your answer.
5. **Get Results**: After completing the set number of questions, navigate to the **Evaluation** and **Analytics** tabs for your full roadmap.

## 👨‍💻 Developed By

**Apurba Roy**
- [LinkedIn](https://linkedin.com/in/apurba-roy05)
- [GitHub](https://github.com/coder-apr-5)
- [Email](mailto:apurbaroy.leo5@gmail.com)

---
© 2026 AI Interview Coach • Built with Gradio & Groq
