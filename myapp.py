import plotly.graph_objects as go
import json
import re
import base64

from groq import Groq

from gtts import gTTS
from faster_whisper import WhisperModel

import PyPDF2
import time

import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()

# --- Lazy LLM initialization ---
llm_client = None

def get_llm():
    """Returns the initialized Groq client."""
    global llm_client
    if llm_client: return llm_client
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key: raise gr.Error("GROQ_API_KEY is missing!")
    llm_client = Groq(api_key=api_key)
    return llm_client

# --- Global Model Loading (Avoid First-Request Delay) ---
WHISPER_MODEL = None
print("Pre-loading Whisper model (tiny)...")
try:
    WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("Whisper ready.")
except Exception as e:
    print(f"Whisper preload failed: {e}")

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
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
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
    prompt = f"""
    Analyze if this text is actually a candidate's resume/CV. 
    If YES, provide a 2 sentence summary of strengths.
    If NO (e.g., it's a random document, book, or nonsense), respond ONLY with the word 'INVALID'.
    
    Text: {resume_text[:2000]}
    """
    response = chat_with_llm("Technical Resume Gatekeeper", prompt)
    return response.strip()

def Job_Description_Expert(job_desc):
    prompt = f"""
    Analyze if this text is a valid Job Description with role details.
    If YES, provide a 2 sentence summary of requirements.
    If NO (e.g., random text, single word, or irrelevant), respond ONLY with the word 'INVALID'.
    
    Text: {job_desc[:2000]}
    """
    response = chat_with_llm("Job Requirement Gatekeeper", prompt)
    return response.strip()

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
    Based on the following interview history and job requirements, provide a professional evaluation.
    Format your response purely in Markdown. Use these specific color spans for categorization:
    - For Strengths: <span style='color: #92fe9d; font-weight: bold;'>[STRENGTH]</span>
    - For Weaknesses: <span style='color: #ff4b4b; font-weight: bold;'>[WEAKNESS]</span>
    - For Areas to Improve/Not Ready: <span style='color: #ffcc00; font-weight: bold;'>[NOT READY YET]</span>
    
    Structure the report with clear headings, bullet points, and ample white space.
    Return a JSON object with these keys: 
    1. "text_evaluation": The full formatted Markdown report.
    2. "correction_needed": A detailed and comprehensive list of specific improvement points and fixes (as a JSON array of strings).
    3. "spoken_conclusion": A short, 2-3 sentence concluding verbal remark to the candidate summarized from the evaluation. Be professional, direct, and mention if the performance was satisfactory or requires significant work. End with a thank you. No emotions.
    4. "scores": {{"Communication": x, "Technical Skills": x, "Problem Solving": x, "Confidence": x, "Cultural Fit": x}} 
    5. "benchmarks": {{"Communication": y, "Technical Skills": y, "Problem Solving": y, "Confidence": y, "Cultural Fit": y}}
    
    GUIDELINE: Benchmarks should represent a high-performing (Top 10%) professional for the role. These values should typically range between 75 and 90 to provide a realistic challenge and standard.

    Interview History: {chat_histories}
    Job Summary: {job_summary}
    """
    response_json = chat_with_llm("Senior HR Evaluator. Output JSON.", prompt, json_mode=True)
    try:
        data = json.loads(response_json)
        # Add extra spacing and clear sections
        eval_text = data.get('text_evaluation', "")
        
        # Ensure clear separation with horizontal lines and double spacing
        eval_text = eval_text.replace("###", "\n---\n###")
        eval_text = eval_text.replace("##", "\n---\n##")
        eval_text = eval_text.replace("\n*", "\n\n*") # Extra space for bullet points
        
        # Add Spoken Conclusion to the Top
        conclusion = data.get('spoken_conclusion', '')
        if conclusion:
            eval_text = f"## 🎤 Final HR Verdict\n**{conclusion}**\n\n---\n\n" + eval_text
            
        # Append real-time Q&A transcript to the basis of the evaluation
        qna_transcript = "\n---\n## 📝 Q&A Transcript\n\n"
        for q, a in chat_histories.items():
            qna_transcript += f"**🧔 Interviewer:** {q}\n\n**🎙️ You:** {a}\n\n"
            
        data['text_evaluation'] = eval_text + qna_transcript
        
        # Format corrections as bullet points if they are in a list
        corrections = data.get('correction_needed', "")
        if isinstance(corrections, list):
            data['correction_needed'] = "\n".join([f"- {c}" for c in corrections])
        elif isinstance(corrections, str) and corrections.strip():
            # If it's a string, ensure it's treated as markdown bullets or add them
            if not corrections.strip().startswith(("-", "*", "1.")):
                data['correction_needed'] = "- " + corrections.replace("\n", "\n- ")

        return data
    except:
        return {
            "text_evaluation": "### Evaluation unavailable. \nPlease try again.",
            "correction_needed": "* No data available.",
            "scores": {"Communication": 70, "Technical Skills": 70, "Problem Solving": 70, "Confidence": 70, "Cultural Fit": 70},
            "benchmarks": {"Communication": 75, "Technical Skills": 75, "Problem Solving": 75, "Confidence": 75, "Cultural Fit": 75}
        }

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
        b_values = [benchmarks.get(cat, 75) for cat in categories]
        fig_radar.add_trace(go.Scatterpolar(
            r=b_values,
            theta=categories,
            fill='toself',
            name='Industry Benchmark',
            line_color='#2eb82e',
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
        go.Bar(name='Benchmark', x=categories, y=[benchmarks.get(c, 75) for c in categories], marker_color='#2eb82e')
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
def transcribe_audio_faster_whisper(audio_path):
    global WHISPER_MODEL
    if not audio_path or not os.path.exists(audio_path): return ""
    
    # Fallback if preload failed
    if WHISPER_MODEL is None:
        try:
            WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
        except:
            return "Speech recognition error."
    
    try:
        segments, info = WHISPER_MODEL.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def text_to_speech(text):
    print(f"Generating TTS for: {text[:50]}...")
    tts = gTTS(text=text, lang='en')
    output_path = "temp_voice.mp3"
    tts.save(output_path)
    return output_path

    return output_path

# --- Application Flow ---
def next_question(resume_pdf, job_desc, num_q, interviewer_audio, user_audio, chat_histories, interview_step, resume_summary, job_summary, latest_question_text):
    print(f"\n🚀 [EVENT] Button Clicked - Step: {interview_step}")
    print(f"DEBUG: resume_pdf_type={type(resume_pdf)}, job_desc_len={len(job_desc) if job_desc else 0}")
    
    # Handle Gradio 5 file list behavior
    if isinstance(resume_pdf, list) and len(resume_pdf) > 0:
        resume_pdf = resume_pdf[0]
        print(f"DEBUG: Extracted first file from list: {resume_pdf}")

    # 1. Initialize summaries on first step
    if interview_step == 0:
        print("Initializing session...")
        if not resume_pdf:
            print("❌ Failure: No Resume Data Received")
            gr.Warning("⚠️ Resume file missing. Please re-upload your PDF.")
            return (None, gr.update(), "⚠️ Please upload your resume first.", None, None, gr.update(), chat_histories, interview_step, resume_summary, job_summary, latest_question_text)
        if not job_desc or len(job_desc.strip()) < 10:
            print("❌ Failure: Invalid/Empty Job Description")
            gr.Warning("⚠️ Job description is too short or empty.")
            return (None, gr.update(), "⚠️ Job description is too short.", None, None, gr.update(), chat_histories, interview_step, resume_summary, job_summary, latest_question_text)
            
        print(f"Validating resume at: {resume_pdf}")
        try:
            resume_text = extract_text_from_pdf(resume_pdf)
            print(f"Extracted {len(resume_text)} characters from PDF.")
        except Exception as e:
            print(f"❌ PDF Extraction Error: {e}")
            gr.Warning(f"❌ Error reading PDF: {str(e)}")
            return (None, gr.update(), f"❌ Error reading PDF: {str(e)}", None, None, gr.update(), chat_histories, interview_step, resume_summary, job_summary, latest_question_text)
            
        r_summary = Resume_Analyst(resume_text)
        if "INVALID" in r_summary.upper():
            gr.Warning("❌ Access Denied: The uploaded file does not appear to be a valid Resume/CV.")
            return (None, gr.update(), "❌ Invalid Resume.", None, None, gr.update(), chat_histories, interview_step, resume_summary, job_summary, latest_question_text)
            
        j_summary = Job_Description_Expert(job_desc)
        if "INVALID" in j_summary.upper():
            gr.Warning("❌ Access Denied: The Job Description provided is invalid or too brief.")
            return (None, gr.update(), "❌ Invalid Job Description.", None, None, gr.update(), chat_histories, interview_step, resume_summary, job_summary, latest_question_text)
            
        print("Initialization successful.")
        resume_summary = r_summary
        job_summary = j_summary
        chat_histories = {}
        
    # 2. Process user's answer from previous question (if any)
    if user_audio and latest_question_text:
        print(f"User answered. Path: {user_audio}")
        answer_text = transcribe_audio_faster_whisper(user_audio)
        print(f"Transcribed: {answer_text}")
        chat_histories[latest_question_text] = answer_text
    
    # Check if interview is complete
    if interview_step >= num_q:
        print(f"Interview complete after {interview_step} questions. Evaluating...")
        eval_data = Evaluator(chat_histories, job_summary)
        print("Evaluation received.")
        radar, bar = create_performance_charts(eval_data['scores'], eval_data['benchmarks'])
        print("Charts created.")
        
        correction_text = f"### 💡 Correction Needed:\n{eval_data.get('correction_needed', 'Continue practicing to improve your scores.')}"
        
        # 4. Generate Spoken Conclusion
        conclusion_text = eval_data.get('spoken_conclusion', "The interview is now complete. Thank you for your time.")
        conclusion_audio = text_to_speech(conclusion_text)

        return (conclusion_audio, gr.update(value="✅ Interview Complete", interactive=False), 
                eval_data['text_evaluation'], radar, bar, correction_text,
                chat_histories, interview_step + 1, resume_summary, job_summary, "")

    # 3. Generate and speak next question
    print("Generating next question...")
    question = Interviewer(chat_histories, resume_summary, job_summary)
    audio_file = text_to_speech(question)
    
    button_label = f"Submit Answer & Next ({interview_step + 1}/{num_q})"
    
    return (audio_file, gr.update(value=button_label, interactive=True), 
            "Evaluation will appear when the interview ends.", None, None, "",
            chat_histories, interview_step + 1, resume_summary, job_summary, question)

# --- Global Data for Viewer Count ---
VISITOR_SESSIONS = set()

def get_visitor_count():
    return f"<div class='visitor-count'>👥 Viewers: {max(1, len(VISITOR_SESSIONS))}</div>"

def track_visitor(request: gr.Request):
    # This isn't perfect for real-time but works for session tracking in Gradio
    if request:
        session_id = request.session_hash
        VISITOR_SESSIONS.add(session_id)
    return get_visitor_count()

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
console.log("🚀 AI Coach UI Logic Initializing...");

window.toggleFeedback = function() {
    const p = document.getElementById('feedback-panel');
    if (p) p.style.display = (p.style.display === 'none' || p.style.display === '') ? 'flex' : 'none';
};

window.toggleFAQ = function() {
    const faq = document.getElementById('faq-chatbot');
    if (faq) faq.style.display = (faq.style.display === 'none' || faq.style.display === '') ? 'flex' : 'none';
};

window.closeFAQ = function() {
    const faq = document.getElementById('faq-chatbot');
    if (faq) faq.style.display = 'none';
};

window.showAnswer = function(qId) {
    const answers = {
        1: "I analyze your resume and job description to create tailored questions that simulate a real interview experience.",
        2: "I use Groq-powered LLaMA 3.3 for intelligence and Faster-Whisper for high-speed voice recognition.",
        3: "Absolutely. I process your data in real-time and never store your documents or audio on any server.",
        4: "Complete the interview (all questions) and then check the 'Analytics' tab for your detailed performance breakdown.",
        5: "For the best experience, provide a clear job description including Job Title, Key Responsibilities, and Required Skills (Technical & Tools) to help the AI generate relevant questions."
    };
    const display = document.getElementById('faq-answer-display');
    if (!display) return;
    
    if (window.currentFAQId === qId && display.style.display === 'block') {
        display.style.opacity = '0';
        setTimeout(() => { display.style.display = 'none'; }, 300);
        window.currentFAQId = null;
    } else {
        display.innerText = answers[qId];
        display.style.display = 'block';
        setTimeout(() => { display.style.opacity = '1'; }, 10);
        window.currentFAQId = qId;
    }
};

window.interviewTime = 0;
window.interviewTimer = null;
window.startInterviewTimer = function() {
    if (!window.interviewTimer) {
        window.interviewTimer = setInterval(() => {
            window.interviewTime++;
            const m = Math.floor(window.interviewTime / 60).toString().padStart(2, '0');
            const s = (window.interviewTime % 60).toString().padStart(2, '0');
            const el = document.getElementById('interview-timer-display');
            if(el) {
                el.innerHTML = `⏱️ Elapsed: <span style="color:#00d2ff">${m}:${s}</span>`;
            }
        }, 1000);
    }
    return [];
};

setInterval(() => {
    if (!document.body.classList.contains('dark')) document.body.classList.add('dark');
    
    if (typeof window.hr_tips === 'undefined') {
        window.hr_tips = ["Hi, I'm your Personalized Interview Coach", "How can I help you?"];
        window.hr_tip_index = 0;
    }
    
    const hr = document.getElementById('hr-character');
    const speech = document.getElementById('speech-bubble');
    if (hr && speech && !hr.dataset.rdy) {
        hr.onmouseenter = () => {
            speech.innerText = window.hr_tips[window.hr_tip_index];
            window.hr_tip_index = (window.hr_tip_index + 1) % window.hr_tips.length;
            speech.style.opacity = '1';
        };
        hr.onmouseleave = () => speech.style.opacity = '0';
        hr.dataset.rdy = "true";
    }
    
    const form = document.getElementById('feedback-form-element');
    const status = document.getElementById('feedback-status');
    if (form && !form.dataset.rdy) {
        form.onsubmit = async (e) => {
            e.preventDefault();
            status.innerText = "Sending...";
            try {
                const r = await fetch(form.action, { method: 'POST', body: new FormData(form), headers: {'Accept': 'application/json'} });
                status.innerText = r.ok ? "✅ Sent!" : "❌ Error";
                if (r.ok) form.reset();
            } catch { status.innerText = "❌ Failed"; }
            setTimeout(() => status.innerText = "", 3000);
        };
        form.dataset.rdy = "true";
    }

    const footer = document.querySelector('footer:not(.custom-app-footer)');
    if (footer && window.location.hostname.includes('gradio.live')) {
        footer.style.display = 'none';
    }
}, 1000);
"""

custom_css = """
/* Non-blocking Decorative Splash */
#splash-overlay {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: #020202;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    z-index: 100005;
    animation: fadeAway 0.8s 2s forwards;
    pointer-events: none;
}
@keyframes fadeAway { 
    from { opacity: 1; visibility: visible; }
    to { opacity: 0; visibility: hidden; pointer-events: none; height: 0; overflow: hidden; display: none; } 
}

#splash-logo {
    width: 280px; height: auto;
    animation: pulseGlow 2s infinite alternate;
}
@keyframes pulseGlow { from { filter: drop-shadow(0 0 5px #00d2ff); } to { filter: drop-shadow(0 0 25px #3a7bd5); } }
@keyframes fadeIn { to { opacity: 1; visibility: visible; } }

#main-app-content {
    display: block !important;
    padding-top: 10px !important;
}

/* Header Spacing */
.header-container {
    text-align: center;
    margin-bottom: 40px;
}

.main-title {
    font-size: 4.2rem;
    font-weight: 950;
    background: linear-gradient(135deg, #00d2ff, #92fe9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -3px;
    margin: 0;
}

#sub-title {
    text-align: center;
    margin-top: -10px !important;
    margin-bottom: 50px !important;
    font-size: 1.4rem;
    opacity: 0.8;
}

/* Section Spacing - ENSURING CLEAR GAP */
.tabs-container {
    margin-top: 100px !important; /* BIG CLEAR GAP between inputs and results */
    border-top: 1px solid rgba(0,210,255,0.2);
    padding-top: 40px;
}

/* Chatbot & HR Styles */
#hr-fixed-wrapper {
    position: fixed;
    bottom: 0;
    right: 40px;
    display: flex;
    align-items: flex-end;
    gap: 15px;
    z-index: 90000;
}
#faq-chatbot {
    display: none;
    flex-direction: column;
    background: rgba(15,15,15,0.98);
    border: 1px solid rgba(0,210,255,0.3);
    border-radius: 25px;
    width: 320px;
    padding: 25px;
    margin-bottom: 110px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.9);
    backdrop-filter: blur(15px);
    max-height: 70vh;
    overflow-y: auto;
}
.chat-title {
    color: #00d2ff;
    font-weight: 900;
    margin-bottom: 20px;
    font-size: 1.3rem;
}
.faq-btn {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    color: #fff;
    padding: 14px;
    border-radius: 12px;
     margin-bottom: 12px;
    font-size: 0.95rem;
    text-align: left;
    cursor: pointer;
    transition: all 0.3s;
}
.faq-btn:hover {
    background: #00d2ff;
    color: #000;
    transform: translateX(8px);
    font-weight: bold;
}
#faq-answer-display {
    margin-top: 15px;
    font-size: 0.95rem;
    color: #eee;
    background: rgba(0,210,255,0.1);
    padding: 18px;
    border-radius: 15px;
    opacity: 0;
    display: none;
    border-left: 6px solid #00d2ff;
    line-height: 1.5;
    transition: opacity 0.3s ease;
}
#hr-container {
    width: 230px;
    cursor: pointer;
    position: relative;
}
#hr-character {
    width: 100%;
    filter: drop-shadow(0 0 25px rgba(0,210,255,0.35));
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
#hr-character:hover { transform: translateY(-12px); }
#speech-bubble {
    position: absolute;
    top: -110px;
    right: 15px;
    background: #00d2ff;
    color: #000;
    padding: 16px 24px;
    border-radius: 22px;
    font-size: 15px;
    font-weight: 900;
    width: 230px;
    text-align: center;
    opacity: 0;
    transition: opacity 0.3s ease;
    box-shadow: 0 15px 30px rgba(0,0,0,0.6);
}
.gradio-container { background: #050505 !important; border: none !important; }
.dark .gr-button-primary { background: linear-gradient(135deg, #00d2ff, #92fe9d) !important; color: #000 !important; border: none !important; }
.dark .gr-block, .dark .gr-form, .dark .gr-box { background: #111 !important; border: 1px solid #222 !important; }
.dark .gr-input, .dark .gr-select, .dark .gr-file { background: #1a1a1a !important; color: #fff !important; border: 1px solid #333 !important; }

/* Custom Footer and Developer Credit */
.dark footer:not(.custom-app-footer) { opacity: 0.6; }
.dev-credit { 
    font-size: 0.85rem; 
    margin-top: 12px; 
    opacity: 0.8; 
    font-weight: 500; 
    letter-spacing: 1px;
    background: linear-gradient(135deg, #00d2ff, #92fe9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.social-links {
    margin-top: 20px;
    display: flex;
    justify-content: center;
    gap: 25px;
}
.social-links a {
    color: rgba(255,255,255,0.6);
    text-decoration: none;
    font-size: 1.2rem;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 8px;
}
.social-links a:hover {
    color: #00d2ff;
    transform: translateY(-3px);
    text-shadow: 0 0 10px rgba(0,210,255,0.4);
}
.social-icon {
    font-size: 1.4rem;
}
.visitor-count {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: #92fe9d;
    font-weight: 600;
    opacity: 0.8;
    background: rgba(146, 254, 157, 0.1);
    padding: 8px 15px;
    border-radius: 20px;
    border: 1px solid rgba(146, 254, 157, 0.2);
    width: fit-content;
    margin-top: 40px;
    margin-left: auto;
    margin-right: auto;
}

/* Markdown Feedback Styling */
.evaluation-md-box {
    padding: 30px;
    background: rgba(255,255,255,0.03) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(0,210,255,0.15) !important;
    max-height: 800px;
    overflow-y: auto;
    line-height: 1.9;
    color: #eee;
}
.evaluation-md-box h1, .evaluation-md-box h2, .evaluation-md-box h3 {
    margin-top: 40px !important;
    margin-bottom: 20px !important;
    color: #00d2ff !important;
    border-bottom: 1px solid rgba(0,210,255,0.1);
    padding-bottom: 10px;
}
.evaluation-md-box p {
    margin-bottom: 20px;
}
.evaluation-md-box li {
    margin-bottom: 12px;
}

/* Analytics Plot Responsiveness */
.analytics-responsive-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 20px !important;
}

.analytics-plot {
    margin-bottom: 20px !important;
    min-height: 500px !important;
    flex: 1 !important;
    background: rgba(15,15,15,0.5) !important;
    border-radius: 20px;
    border: 1px solid rgba(0,210,255,0.1);
    padding: 10px;
}

@media (max-width: 1000px) {
    .analytics-responsive-row {
        flex-direction: column !important;
    }
    .analytics-plot {
        min-height: 400px !important;
        width: 100% !important;
    }
    .main-title {
        font-size: 2.8rem !important;
    }
}

/* Feedback Section Styles */
#feedback-wrapper {
    position: fixed;
    top: 40px;
    right: 40px;
    z-index: 100000;
    opacity: 0;
    visibility: hidden;
    animation: fadeIn 0.8s 3s forwards;
}
#feedback-btn {
    background: rgba(0,210,255,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0,210,255,0.3);
    color: #00d2ff;
    padding: 10px 20px;
    border-radius: 30px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s;
}
#feedback-btn:hover {
    background: #00d2ff;
    color: #000;
    box-shadow: 0 0 15px rgba(0,210,255,0.6);
}
#feedback-panel {
    display: none;
    flex-direction: column;
    position: fixed !important;
    top: 95px !important;
    right: 40px !important;
    background: rgba(15,15,15,0.98) !important;
    border: 1px solid rgba(0,210,255,0.3) !important;
    border-radius: 20px !important;
    width: 320px !important;
    padding: 25px !important;
    z-index: 100001 !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8);
    backdrop-filter: blur(20px);
}
.feedback-title {
    color: #00d2ff;
    font-weight: 800;
    margin-bottom: 20px;
    font-size: 1.2rem;
    text-align: center;
}
.feedback-form {
    display: flex;
    flex-direction: column;
    gap: 15px;
}
.feedback-form input, .feedback-form textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    padding: 12px !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
.feedback-form button {
    background: linear-gradient(135deg, #00d2ff, #92fe9d) !important;
    color: black !important;
    border: none !important;
    padding: 12px !important;
    border-radius: 10px !important;
    font-weight: bold !important;
    cursor: pointer !important;
    transition: transform 0.2s !important;
}
.feedback-form button:hover {
    transform: scale(1.02) !important;
}

/* Responsive Mobile Fixes */
@media (max-width: 768px) {
    #feedback-wrapper { top: 20px; right: 20px; }
    #feedback-panel {
        right: 20px !important;
        top: 75px !important;
        width: 280px !important;
    }
    .splash-title-text {
        letter-spacing: 4px;
        font-size: 1.1rem;
    }
    
    #hr-fixed-wrapper {
        right: 20px !important;
        bottom: 20px !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        gap: 10px !important;
    }
    
    #hr-container {
        width: 80px !important;
        height: 80px !important;
        border-radius: 50% !important;
        overflow: hidden !important;
        border: 3px solid #00d2ff !important;
        background: #050505 !important;
        box-shadow: 0 0 20px rgba(0,210,255,0.5) !important;
    }
    
    #hr-character {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        object-position: center 10% !important;
    }
    
    #speech-bubble {
        display: none !important;
    }
    
    #faq-chatbot {
        width: 280px !important;
        margin-bottom: 0 !important;
        position: relative !important;
        right: 0 !important;
        padding: 15px !important;
        max-height: 70vh;
        overflow-y: auto;
    }
    
    .main-title {
        font-size: 2.5rem !important;
    }
    
    .faq-btn {
        padding: 10px !important;
        font-size: 0.85rem !important;
    }
}
"""

# Encode images
base_dir = os.path.dirname(os.path.abspath(__file__))
logo_file = os.path.join(base_dir, "logo.png")
hr_file = os.path.join(base_dir, "hr_guy.png")
logo_base64 = get_image_base64(logo_file)
hr_base64 = get_image_base64(hr_file)

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, head=f"<script>\n{custom_js}\n</script>") as demo:
    chat_histories_state = gr.State({})
    interview_step_state = gr.State(0)
    resume_summary_state = gr.State(None)
    job_summary_state = gr.State(None)
    latest_question_text_state = gr.State("")
    # 1. Non-blocking Splash Transition
    gr.HTML(f"""
        <div id="splash-overlay">
            <img id="splash-logo" src="{logo_base64}">
            <div style="color: #fff; margin-top: 25px; letter-spacing: 5px; font-weight: 300;">INITIALIZING HR COACH...</div>
        </div>
    """)

    # 2. Main App Container
    with gr.Column(elem_id="main-app-content"):
        # Feedback Section (Moved inside to only show with main app)
        gr.HTML(f"""
            <div id="feedback-wrapper">
                <button id="feedback-btn" onclick="toggleFeedback()">💬 Feedback</button>
                <div id="feedback-panel">
                    <div class="feedback-title">Share Your Thoughts</div>
                    <form id="feedback-form-element" class="feedback-form" action="https://formspree.io/f/xreyyoqg" method="POST">
                        <input type="text" name="name" placeholder="Your Name" required>
                        <input type="email" name="email" placeholder="Your Email" required>
                        <textarea name="feedback" placeholder="Your Feedback..." rows="4" required></textarea>
                        <button type="submit">Send Message</button>
                    </form>
                    <div id="feedback-status" style="margin-top: 15px; font-weight: 600; text-align: center; min-height: 20px;"></div>
                </div>
            </div>
        """)
        
        # Header Section
        gr.HTML(f"""
            <div class="header-container">
                <h1 class="main-title">AI Interview Coach</h1>
            </div>
        """)
        
        gr.Markdown("### 🧔 Elevate Your Career with Next-Gen AI Feedback", elem_id="sub-title")
        
        with gr.Row():
            with gr.Column():
                resume_input = gr.File(label="📄 Upload Resume (PDF)", type='filepath')
                job_desc_input = gr.Textbox(label="💼 Job Description", lines=10, placeholder="Paste the job requirements here...")
                with gr.Row():
                    num_q_input = gr.Slider(label="❓ Questions", minimum=1, maximum=10, value=5, step=1)
                    timer_display = gr.HTML("<div id='interview-timer-display' style='font-size: 1.1rem; font-weight: bold; color: #92fe9d; margin-top: 30px; text-align: center; background: rgba(0,210,255,0.05); padding: 10px; border-radius: 10px; border: 1px solid rgba(0,210,255,0.2);'>⏳ Est. Time: 10 mins</div>")
                
                # Dynamic update of estimated time based on question count slider
                def update_est(val):
                    return f"<div id='interview-timer-display' style='font-size: 1.1rem; font-weight: bold; color: #92fe9d; margin-top: 30px; text-align: center; background: rgba(0,210,255,0.05); padding: 10px; border-radius: 10px; border: 1px solid rgba(0,210,255,0.2);'>⏳ Est. Time: {val * 2} mins</div>"
                num_q_input.change(fn=update_est, inputs=num_q_input, outputs=timer_display)

                start_btn = gr.Button("🚀 Start Interview", variant="primary", scale=2, elem_id="start-interview-btn")
            
            with gr.Column():
                interviewer_question = gr.Audio(label="🧔 Interviewer Speaks:", type="filepath", interactive=False)
                user_answer = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Your Answer")
                
        # Separation for Evaluation and Analytics with explicit class for spacing
        with gr.Tabs(elem_classes="tabs-container") as tabs:
            with gr.Tab("📝 Detailed Evaluation"):
                gr.HTML("<div style='margin-bottom: 20px; font-weight: bold; color: #00d2ff; text-transform: uppercase; letter-spacing: 2px;'>HR Feedback & Roadmap</div>")
                evaluation_textbox = gr.Markdown("The evaluation results will appear here after the interview ends.", elem_classes="evaluation-md-box")
            with gr.Tab("📊 Performance Analytics"):
                with gr.Row(elem_classes="analytics-responsive-row"):
                    radar_plot = gr.Plot(label="Skill Competency", elem_classes="analytics-plot")
                    bar_plot = gr.Plot(label="Peer Benchmarks", elem_classes="analytics-plot")
                correction_md = gr.Markdown("### 💡 Correction Needed\nYour analysis will appear here after the interview is complete.", elem_id="correction-needed-md")

    # 3. Interactive HR Character Overlay
    gr.HTML(f"""
        <div id="hr-fixed-wrapper">
            <div id="faq-chatbot">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <div class="chat-title" style="margin-bottom: 0;">🧔 Assistant Coach</div>
                    <button id="close-faq" onclick="closeFAQ()" style="background: none; border: none; color: #00d2ff; font-size: 28px; cursor: pointer; line-height: 1;">&times;</button>
                </div>
                <button class="faq-btn" onclick="showAnswer(1)">❓ How does it work?</button>
                <button class="faq-btn" onclick="showAnswer(2)">❓ AI Models used?</button>
                <button class="faq-btn" onclick="showAnswer(3)">❓ Data security?</button>
                <button class="faq-btn" onclick="showAnswer(4)">❓ Where are results?</button>
                <button class="faq-btn" onclick="showAnswer(5)">❓ How to write the JD?</button>
                <div id="faq-answer-display"></div>
            </div>
            <div id="hr-container" onclick="toggleFAQ()">
                <div id="speech-bubble">Hi, I'm your Personalized Interview Coach</div>
                <div id="hr-character">
                    <img src="{hr_base64}" alt="HR Coach">
                </div>
            </div>
        </div>
    """)

    # 4. Custom Footer & Visitor Counter
    with gr.Column(elem_id="footer-area"):
        visitor_md = gr.HTML(get_visitor_count(), elem_id="visitor-wrapper")
        gr.HTML("""
            <footer class="custom-app-footer" style="text-align: center; padding: 40px 20px; color: rgba(255,255,255,0.5);">
                <p style="font-size: 0.9rem;">© 2026 AI Interview Coach • Built with Gradio & Groq • Elevate Your Career</p>
                <div class="dev-credit">Developed by Apurba Roy</div>
                <div class="social-links">
                    <a href="https://linkedin.com/in/apurba-roy05" target="_blank" title="LinkedIn">
                        <span class="social-icon">🔗</span> LinkedIn
                    </a>
                    <a href="https://github.com/coder-apr-5/interview_coach" target="_blank" title="GitHub">
                        <span class="social-icon">💻</span> GitHub
                    </a>
                    <a href="mailto:apurbaroy.leo5@gmail.com" title="Email">
                        <span class="social-icon">✉️</span> Mail
                    </a>
                </div>
            </footer>
        """)

    # 5. Periodic visitor update
    demo.load(track_visitor, None, visitor_md)

    start_btn.click(
        fn=next_question,
        inputs=[resume_input, job_desc_input, num_q_input, interviewer_question, user_answer, chat_histories_state, interview_step_state, resume_summary_state, job_summary_state, latest_question_text_state],
        outputs=[interviewer_question, start_btn, evaluation_textbox, radar_plot, bar_plot, correction_md, chat_histories_state, interview_step_state, resume_summary_state, job_summary_state, latest_question_text_state],
        js="function() { if(window.startInterviewTimer) { window.startInterviewTimer(); } return arguments; }"
    ).then(
        fn=lambda: None,
        inputs=[],
        outputs=[user_answer]
    )

    # 6. Final cleanup (HTML script injection removed because we use 'head' arg in blocks now)

if __name__ == "__main__":
    demo.launch(share=True, show_api=False)
