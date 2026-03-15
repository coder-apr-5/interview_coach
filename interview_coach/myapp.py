import plotly.graph_objects as go
import json
import re
import base64

from groq import Groq

from gtts import gTTS
from faster_whisper import WhisperModel

import time

import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()

# --- Lazy LLM initialization ---
llm_client = None

def get_llm():
    """Lazily initialize the Groq client on first use."""
    global llm_client
    if llm_client is not None:
        return llm_client

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise gr.Error("⚠️ GROQ_API_KEY is not set. Please add it to your .env file.")
    
    llm_client = Groq(api_key=api_key)
    return llm_client

def chat_with_llm(role, content, json_mode=False):
    client = get_llm()
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": content}
    ]
    
    response_format = {"type": "json_object"} if json_mode else None
    
    for attempt in range(3):
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                response_format=response_format
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < 2:
                time.sleep(2)
                continue
            return f"Error: {str(e)}"

# --- NLP / Interview Logic ---
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return text

def Resume_Analyst(resume_text):
    prompt = f"Analyze this resume and provide a 2 sentence summary of strengths and core skills: {resume_text}"
    return chat_with_llm("Technical Resume Expert", prompt)

def Job_Description_Expert(job_desc):
    prompt = f"Summarize the key requirements and technologies for this job description in 2 sentences: {job_desc}"
    return chat_with_llm("Job Requirement Analyst", prompt)

def Interviewer(chat_histories, resume_summary, job_summary):
    prompt = f"""
    Context:
    Resume: {resume_summary}
    Job: {job_summary}
    History: {chat_histories}
    
    Act as a professional interviewer. Ask ONE insightful follow-up question based on the history or the candidate's profile. 
    Keep it conversational and professional.
    """
    return chat_with_llm("Professional Interviewer", prompt)

def Evaluator(chat_histories, job_summary):
    prompt = f"""
    Based on the following interview history and job requirements, provide:
    1. "text_evaluation": A detailed point-by-point critique and a specific correction plan for the candidate's weaknesses.
    2. "scores": An object with scores 0-100 for: Communication, Technical Skills, Problem Solving, Confidence, and Cultural Fit.
    3. "benchmarks": Average scores for this role for comparison.

    Answer Histories: {chat_histories}
    Job Summary: {job_summary}
    """
    response_json = chat_with_llm("Senior HR Evaluator. Output JSON.", prompt, json_mode=True)
    try:
        data = json.loads(response_json)
        return data
    except:
        return {"text_evaluation": response_json, "scores": {"Communication": 70, "Technical Skills": 70, "Problem Solving": 70, "Confidence": 70, "Cultural Fit": 70}, "benchmarks": {"Communication": 75, "Technical Skills": 75, "Problem Solving": 75, "Confidence": 75, "Cultural Fit": 75}}

def create_performance_charts(scores, benchmarks=None):
    # Radar Chart
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Performance',
        line_color='#00d1ff'
    ))
    
    if benchmarks:
        b_values = [benchmarks.get(cat, 50) for cat in categories]
        fig_radar.add_trace(go.Scatterpolar(
            r=b_values,
            theta=categories,
            fill='toself',
            name='Industry Benchmark',
            line_color='#92fe9d',
            opacity=0.5
        ))
        
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title="Skills Radar"
    )

    # Bar Chart Comparison
    fig_bar = go.Figure(data=[
        go.Bar(name='You', x=categories, y=values, marker_color='#00d1ff'),
        go.Bar(name='Benchmark', x=categories, y=[benchmarks.get(c, 50) for c in categories], marker_color='#92fe9d')
    ])
    fig_bar.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title="Direct Comparison"
    )
    
    return fig_radar, fig_bar

# --- Voice / Audio ---
WHISPER_MODEL = None

def transcribe_audio_faster_whisper(audio_path):
    global WHISPER_MODEL
    if audio_path is None: return ""
    
    print(f"Transcribing audio: {audio_path}")
    if WHISPER_MODEL is None:
        print("Loading Whisper model...")
        WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    try:
        segments, info = WHISPER_MODEL.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        print(f"Transcription result: {text}")
        return text
    except Exception as e:
        print(f"Transcription error: {e}")
        return "Audio transcription failed."

def text_to_speech(text):
    print(f"Generating TTS for: {text[:50]}...")
    tts = gTTS(text=text, lang='en')
    output_path = "temp_voice.mp3"
    tts.save(output_path)
    return output_path

# --- Application Flow ---
def next_question(resume_pdf, job_desc, num_q, interviewer_audio, user_audio, chat_histories, interview_step, resume_summary, job_summary, latest_question_text):
    print(f"\n--- STEP {interview_step + 1} / {num_q} ---")
    
    # 1. Initialize summaries on first step
    if interview_step == 0:
        print("Initializing session...")
        resume_text = extract_text_from_pdf(resume_pdf)
        resume_summary = Resume_Analyst(resume_text)
        job_summary = Job_Description_Expert(job_desc)
        chat_histories = {}
        
    # 2. Process user's answer from previous question (if any)
    if user_audio and latest_question_text:
        answer_text = transcribe_audio_faster_whisper(user_audio)
        chat_histories[latest_question_text] = answer_text
    
    # Check if interview is complete
    if interview_step >= num_q:
        print("Interview complete. Generating evaluation...")
        eval_data = Evaluator(chat_histories, job_summary)
        radar, bar = create_performance_charts(eval_data['scores'], eval_data['benchmarks'])
        return None, None, gr.update(value="Interview Ended", interactive=False), eval_data['text_evaluation'], radar, bar, chat_histories, interview_step + 1, resume_summary, job_summary, ""

    # 3. Generate and speak next question
    print("Generating next question...")
    question = Interviewer(chat_histories, resume_summary, job_summary)
    audio_file = text_to_speech(question)
    
    return audio_file, None, gr.update(interactive=False), "Evaluation will appear when the interview ends.", None, None, chat_histories, interview_step + 1, resume_summary, job_summary, question

def get_image_base64(image_path):
    """Convert an image file to a base64 string for embedding in HTML."""
    try:
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return ""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        print(f"❌ Error encoding image {image_path}: {e}")
        return ""

custom_js = """
function initApp() {
    console.log("Initializing Premium AI Interview Coach Experience...");
    
    const splash = document.getElementById('splash-screen');
    const mainApp = document.getElementById('main-app-content');
    
    // Smooth transition logic
    setTimeout(() => {
        if (splash) {
            splash.classList.add('splash-blur-exit');
            setTimeout(() => {
                splash.style.display = 'none';
                if (mainApp) {
                    mainApp.style.display = 'block';
                    setTimeout(() => mainApp.style.opacity = '1', 50);
                }
                initHR(); 
            }, 1200);
        }
    }, 3800);
}

function initHR() {
    const container = document.getElementById('hr-character');
    const bubble = document.getElementById('speech-bubble');
    const chatbox = document.getElementById('faq-chatbot');
    
    if (!container || !bubble) return;

    let greetings = [
        "Hi, I'm your Personalized Interview Coach",
        "How can I help you?"
    ];
    let currentIdx = 0;
    let lastHoverTime = 0;

    container.addEventListener('mouseenter', () => {
        const now = Date.now();
        if (now - lastHoverTime > 500) {
            bubble.innerText = greetings[currentIdx];
            currentIdx = (currentIdx + 1) % greetings.length;
            bubble.style.opacity = '1';
            lastHoverTime = now;
        }
    });

    container.addEventListener('mouseleave', () => {
        bubble.style.opacity = '0';
    });

    container.addEventListener('click', (e) => {
        e.stopPropagation();
        if (chatbox.style.display === "none" || chatbox.style.display === "") {
            chatbox.style.display = "flex";
        } else {
            chatbox.style.display = "none";
        }
    });

    window.showAnswer = function(questionId) {
        const answers = {
            1: "I analyze your resume and job description to create tailored questions that simulate a real interview experience.",
            2: "I use Groq-powered LLaMA 3.3 for intelligence and Faster-Whisper for high-speed voice recognition.",
            3: "Absolutely. I process your data in real-time and never store your documents or audio on any server.",
            4: "Complete the interview (all questions) and then check the 'Analytics' tab for your detailed performance breakdown."
        };
        const display = document.getElementById('faq-answer-display');
        display.innerText = answers[questionId];
        display.style.opacity = '1';
    };
}

const interval = setInterval(() => {
    if (document.getElementById('splash-screen')) {
        initApp();
        clearInterval(interval);
    }
}, 500);
"""

custom_css = """
/* Premium Splash Screen */
#splash-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: #020202;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 99999;
}
#splash-logo {
    width: 300px;
    height: auto;
    filter: blur(20px);
    opacity: 0;
    animation: blurToFocus 2.5s ease-out forwards, pulseGlow 3s infinite alternate;
}
@keyframes blurToFocus {
    0% { filter: blur(30px); opacity: 0; transform: scale(0.85); }
    50% { filter: blur(10px); opacity: 0.5; }
    100% { filter: blur(0); opacity: 1; transform: scale(1); }
}
@keyframes pulseGlow {
    from { filter: drop-shadow(0 0 10px rgba(0,210,255,0.2)); }
    to { filter: drop-shadow(0 0 30px rgba(0,210,255,0.6)); }
}
.splash-blur-exit {
    transition: all 1.2s cubic-bezier(0.645, 0.045, 0.355, 1);
    filter: blur(50px);
    opacity: 0;
}
.splash-title-text {
    margin-top: 30px;
    color: #fff;
    font-family: 'Inter', sans-serif;
    letter-spacing: 8px;
    text-transform: uppercase;
    font-size: 1.2rem;
    opacity: 0;
    animation: fadeIn 1.5s 1s forwards;
}
@keyframes fadeIn { to { opacity: 0.8; } }

/* Main App Layout */
#main-app-content {
    display: none;
    opacity: 0;
    transition: opacity 1s ease-in;
}

/* Chatbot & HR Styles */
#hr-fixed-wrapper {
    position: fixed;
    bottom: 0;
    right: 30px;
    display: flex;
    align-items: flex-end;
    gap: 15px;
    z-index: 90000;
}
#faq-chatbot {
    display: none;
    flex-direction: column;
    background: #111;
    border: 1px solid rgba(0,210,255,0.4);
    border-radius: 20px;
    width: 320px;
    padding: 20px;
    margin-bottom: 100px;
    box-shadow: 0 15px 50px rgba(0,0,0,0.9);
    backdrop-filter: blur(10px);
}
.chat-title {
    color: #00d2ff;
    font-weight: 900;
    margin-bottom: 15px;
    font-size: 1.2rem;
}
.faq-btn {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #fff;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-size: 0.95rem;
    text-align: left;
    cursor: pointer;
    transition: all 0.3s;
}
.faq-btn:hover {
    background: #00d2ff;
    color: #000;
    transform: translateX(5px);
}
#faq-answer-display {
    margin-top: 15px;
    font-size: 0.9rem;
    color: #ddd;
    background: rgba(0,210,255,0.1);
    padding: 15px;
    border-radius: 12px;
    opacity: 0;
    border-left: 5px solid #00d2ff;
}
#hr-container {
    width: 240px;
    cursor: pointer;
    position: relative;
}
#hr-character {
    width: 100%;
    filter: drop-shadow(0 0 15px rgba(0,210,255,0.3));
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
#hr-character:hover { transform: translateY(-10px); }
#speech-bubble {
    position: absolute;
    top: -100px;
    right: 15px;
    background: #00d2ff;
    color: #000;
    padding: 15px 20px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 900;
    width: 210px;
    text-align: center;
    opacity: 0;
    transition: opacity 0.3s ease;
    box-shadow: 0 10px 20px rgba(0,0,0,0.5);
}

/* UI Sharpness */
.header-logo-container {
    display: flex;
    align-items: center;
    gap: 30px;
    margin: 50px 0;
}
.header-logo-container img { height: 100px; }
.main-title {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d2ff, #92fe9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -3px;
}
"""

with gr.Blocks() as demo:
    chat_histories_state = gr.State({})
    interview_step_state = gr.State(0)
    resume_summary_state = gr.State(None)
    job_summary_state = gr.State(None)
    latest_question_text_state = gr.State("")

    # Encode images
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_file = os.path.join(base_dir, "logo.png")
    hr_file = os.path.join(base_dir, "hr_guy.png")
    logo_base64 = get_image_base64(logo_file)
    hr_base64 = get_image_base64(hr_file)

    # 1. Splash Screen
    gr.HTML(f"""
        <div id="splash-screen">
            <img id="splash-logo" src="{logo_base64}" alt="AI Coaching">
            <div class="splash-title-text">Preparing Your Session</div>
        </div>
    """)

    # 2. Main App Container
    with gr.Column(elem_id="main-app-content"):
        # Header
        gr.HTML(f"""
            <div class="header-logo-container">
                <img src="{logo_base64}" alt="Logo">
                <h1 class="main-title">AI Interview Coach</h1>
            </div>
        """)
        
        gr.Markdown("### 🧔 Elevate Your Career with Next-Gen AI Feedback")
        
        with gr.Row():
            with gr.Column():
                resume_input = gr.File(label="📄 Upload Resume (PDF)", type='filepath')
                job_desc_input = gr.Textbox(label="💼 Job Description", lines=10, placeholder="Paste the job requirements here...")
                num_q_input = gr.Slider(label="❓ Questions", minimum=1, maximum=10, value=5, step=1)
                start_btn = gr.Button("🚀 Start Interview", variant="primary", scale=2)
            
            with gr.Column():
                interviewer_question = gr.Audio(label="🧔 Interviewer Speaks:", type="filepath", interactive=False)
                user_answer = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Your Answer")
                
        with gr.Tabs() as tabs:
            with gr.Tab("📝 Evaluation"):
                evaluation_textbox = gr.Textbox(label="HR Feedback & Roadmap", lines=15)
            with gr.Tab("📊 Analytics"):
                with gr.Row():
                    radar_plot = gr.Plot(label="Skill Competency")
                    bar_plot = gr.Plot(label="Peer Benchmarks")

    # 3. Interactive HR Character Overlay
    gr.HTML(f"""
        <div id="hr-fixed-wrapper">
            <div id="faq-chatbot">
                <div class="chat-title">🧔 Assistant Coach</div>
                <button class="faq-btn" onclick="showAnswer(1)">❓ How does it work?</button>
                <button class="faq-btn" onclick="showAnswer(2)">❓ AI Models used?</button>
                <button class="faq-btn" onclick="showAnswer(3)">❓ Data security?</button>
                <button class="faq-btn" onclick="showAnswer(4)">❓ Where are results?</button>
                <div id="faq-answer-display">Select a query.</div>
            </div>
            <div id="hr-container">
                <div id="speech-bubble">Hi, I'm your Targeted Interview Coach</div>
                <div id="hr-character">
                    <img src="{hr_base64}" alt="HR Coach">
                </div>
            </div>
        </div>
    """)

    start_btn.click(
        fn=next_question,
        inputs=[resume_input, job_desc_input, num_q_input, interviewer_question, user_answer, chat_histories_state, interview_step_state, resume_summary_state, job_summary_state, latest_question_text_state],
        outputs=[interviewer_question, user_answer, start_btn, evaluation_textbox, radar_plot, bar_plot, chat_histories_state, interview_step_state, resume_summary_state, job_summary_state, latest_question_text_state]
    )

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft(), css=custom_css, js=custom_js)
