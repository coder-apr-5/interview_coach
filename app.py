import plotly.graph_objects as go
import json
import re
import base64
import random

from groq import Groq

from gtts import gTTS
from faster_whisper import WhisperModel

import PyPDF2
import time

import gradio as gr
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse

load_dotenv()

# --- HR Personas (Shuffled each session with matching male/female neural voices) ---
HR_PERSONAS = [
    {"name": "Alex Carter", "gender": "male", "voice": "en-US-ChristopherNeural", "title": "Senior AI HR Consultant"},
    {"name": "Sarah Jenkins", "gender": "female", "voice": "en-US-JennyNeural", "title": "Head of Talent Acquisition"},
    {"name": "David Miller", "gender": "male", "voice": "en-US-GuyNeural", "title": "Lead Technical Recruiter"},
    {"name": "Emily Watson", "gender": "female", "voice": "en-US-AriaNeural", "title": "Senior People & Culture Lead"},
    {"name": "Michael Vance", "gender": "male", "voice": "en-GB-RyanNeural", "title": "Executive Hiring Manager"},
    {"name": "Jessica Taylor", "gender": "female", "voice": "en-GB-SoniaNeural", "title": "Principal Talent Partner"},
    {"name": "Daniel Brooks", "gender": "male", "voice": "en-US-EricNeural", "title": "Engineering Recruitment Director"},
    {"name": "Sophia Martinez", "gender": "female", "voice": "en-US-AvaNeural", "title": "Global Hiring Consultant"},
    {"name": "Marcus Reed", "gender": "male", "voice": "en-AU-WilliamNeural", "title": "Technical Talent Strategist"},
    {"name": "Rachel Adams", "gender": "female", "voice": "en-AU-NatashaNeural", "title": "Senior Technical Recruiter"}
]

# --- Lazy LLM initialization ---
llm_client = None

def get_llm():
    """Returns the initialized Groq client."""
    global llm_client
    if llm_client: return llm_client
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key: raise RuntimeError("❌ GROQ_API_KEY is missing! Please configure the GROQ_API_KEY environment variable in your host settings (e.g., Render Environment Variables or .env file).")
    llm_client = Groq(api_key=api_key)
    return llm_client

# --- Global Model Loading (Lazy) ---
WHISPER_MODEL = None

def get_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("Loading Whisper model (tiny)...")
        try:
            # Explicitly setting device='cpu' and compute_type='int8' for 100% stability on free HF Spaces
            WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
            print("Whisper ready.")
        except Exception as e:
            print(f"Whisper load failed: {e}")
            # Fallback to None if initialization fails to prevent total app crash
    return WHISPER_MODEL

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
                model="openai/gpt-oss-120b",
                response_format=response_format
            )
            res = chat_completion.choices[0].message.content
            if not res or not res.strip():
                continue
            return res
        except Exception as e:
            print(f"❌ Groq Error: {e}")
            if attempt < 2: time.sleep(2)
            else: return f"Error: {str(e)}"
    return "Error: LLM timeout."

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

def Interviewer(chat_histories, resume_summary, job_summary, current_step=1, total_q=5, hr_persona=None):
    chat_histories = chat_histories or {}
    q_count = len(chat_histories) + 1
    
    persona = hr_persona if (hr_persona and isinstance(hr_persona, dict)) else HR_PERSONAS[0]
    hr_name = persona.get("name", "Alex Carter")
    hr_title = persona.get("title", "AI HR Consultant")
    
    if q_count == 1:
        role = f"{hr_name} ({hr_title} - Opening Session)"
        prompt = f"""
        Context:
        Resume Summary: {resume_summary}
        Job Requirements: {job_summary}
        Total Questions: {total_q}
        
        Task:
        Introduce yourself warmly as {hr_name}, a {hr_title}. Based on the candidate's resume and job requirements, ask a strong, relevant FIRST question to start the interview (e.g., asking them to introduce themselves and walk through a key project from their resume).
        Keep it professional, concise, and engaging. Ask ONE question only.
        """
    else:
        role = f"{hr_name} (Question {q_count} of {total_q})"
        prompt = f"""
        Context:
        Resume: {resume_summary}
        Job: {job_summary}
        Previous Interview Q&A History:
        {json.dumps(chat_histories, indent=2)}
        
        Current Question Number: {q_count} of {total_q}
        
        CRITICAL RULES:
        1. DO NOT re-introduce yourself or repeat greetings like "Hello, I'm {hr_name}...". You are already in the middle of the interview.
        2. DO NOT repeat any question previously asked in the History.
        3. Transition naturally from their previous response in 1 short sentence, then ask a NEW question on a DIFFERENT topic.
        
        Topic Roadmap by Question Number:
        - Question 2: Technical Deep Dive (Core programming concepts, frameworks, or tools listed in job & resume)
        - Question 3: Problem Solving & Debugging (Handling tricky bugs, edge cases, or performance trade-offs)
        - Question 4: Behavioral & Teamwork (Handling conflict, tight deadlines, or feedback using STAR method)
        - Question 5+: System Architecture, Security, Scalability, or Scenario Analysis
        
        Ask ONE clear, direct question for Question #{q_count}.
        """
    
    return chat_with_llm(role, prompt)

def Evaluator(chat_histories, job_summary):
    prompt = f"""
    You are a strict, objective Senior HR Evaluator. Evaluate the candidate's actual responses provided in the Interview History below against the Job Requirements.
    
    CRITICAL SCORING RULES:
    1. Score EACH metric strictly based ONLY on what the candidate actually answered in the Interview History:
       - 0 to 30 (Poor/Failed): Empty/missing answer, completely incorrect technical facts, or no effort.
       - 31 to 50 (Weak): Superficial/vague answer, missing key technical concepts, low confidence.
       - 51 to 70 (Average/Fair): Correct basics, but lacks architectural depth or structured examples.
       - 71 to 85 (Good): Thorough, accurate, articulate, and well-reasoned answers.
       - 86 to 100 (Exceptional): Top-tier expert answers with deep system design & trade-off mastery.
       
    2. DO NOT BE OVERLY POLITE OR INFLATE SCORES. If the candidate gave weak, brief, missing, or incorrect responses, their scores MUST be low (e.g. 20 to 55).
    3. The numerical scores MUST match the Strengths, Weaknesses, and Verdict in your text evaluation.

    Return a JSON object with these exact keys: 
    1. "text_evaluation": The full formatted Markdown report. Use these specific color spans for categorization:
       - For Strengths: <span style='color: #92fe9d; font-weight: bold;'>[STRENGTH]</span>
       - For Weaknesses: <span style='color: #ff4b4b; font-weight: bold;'>[WEAKNESS]</span>
       - For Areas to Improve/Not Ready: <span style='color: #ffcc00; font-weight: bold;'>[NOT READY YET]</span>
    2. "correction_needed": A detailed list of specific improvement points and fixes (as a JSON array of strings).
    3. "spoken_conclusion": A short, 2-3 sentence concluding verbal remark to the candidate summarized from the evaluation. Be professional, direct, and mention if the performance was satisfactory or requires significant work. End with a thank you. No emotions.
    4. "scores": {{"Communication": score, "Technical Skills": score, "Problem Solving": score, "Confidence": score, "Cultural Fit": score}} 
    5. "benchmarks": {{"Communication": 80, "Technical Skills": 85, "Problem Solving": 85, "Confidence": 80, "Cultural Fit": 80}}

    Interview History:
    {json.dumps(chat_histories, indent=2)}

    Job Requirements Summary:
    {job_summary}
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
            "scores": {"Communication": 40, "Technical Skills": 40, "Problem Solving": 40, "Confidence": 40, "Cultural Fit": 40},
            "benchmarks": {"Communication": 80, "Technical Skills": 85, "Problem Solving": 85, "Confidence": 80, "Cultural Fit": 80}
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
def resolve_path(obj):
    if obj is None: return None
    if isinstance(obj, str): return obj
    if isinstance(obj, list) and len(obj) > 0: return resolve_path(obj[0])
    if hasattr(obj, 'path'): return obj.path
    if isinstance(obj, dict): return obj.get('path') or obj.get('name')
    try:
        if hasattr(obj, 'name'): return obj.name
    except:
        pass
    return str(obj)

def transcribe_audio_faster_whisper(audio_path):
    audio_path = resolve_path(audio_path)
    if not audio_path or not os.path.exists(audio_path): return ""
    
    model = get_whisper()
    if model is None:
        return "Speech recognition error."
    
    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def text_to_speech(text, voice_name="en-US-ChristopherNeural"):
    import time
    import tempfile
    import asyncio
    
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"voice_{int(time.time()*1000)}.mp3")
    
    # 1. Try high-quality Neural edge-tts voice matching HR gender
    try:
        import edge_tts
        print(f"Generating Edge-TTS ({voice_name}) for: {text[:50]}...")
        asyncio.run(edge_tts.Communicate(text, voice_name).save(output_path))
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception as e:
        print(f"Edge-TTS failed ({e}), falling back to gTTS...")

    # 2. Fallback to gTTS if edge-tts fails
    try:
        print(f"Generating gTTS fallback for: {text[:50]}...")
        tts = gTTS(text=text, lang='en')
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def next_question(resume_pdf, job_desc, num_q, interviewer_audio, user_audio, user_text_ans, chat_histories, interview_step, resume_summary, job_summary, latest_question_text, hr_persona):
    print(f"\n[EVENT] Button Clicked - Current Step: {interview_step}")
    chat_histories = chat_histories or {}
    num_q_val = int(num_q)
    
    # Robust path resolution
    resume_path = resolve_path(resume_pdf)
    user_audio_path = resolve_path(user_audio)
    
    # 1. Initialization (Step 0)
    if interview_step == 0:
        print("Initializing session...")
        if not resume_path:
            gr.Warning("⚠️ Please upload your resume PDF.")
            return (None, gr.update(), "⚠️ Please upload your resume first.", None, None, gr.update(), chat_histories, interview_step, resume_summary, job_summary, "⚠️ Error: Resume missing.", "⚠️ Error: Resume missing.", None, "", None)
        
        if not job_desc or len(job_desc.strip()) < 10:
            gr.Warning("⚠️ Please provide a clear Job Description.")
            return (None, gr.update(), "⚠️ Job description is too short.", None, None, gr.update(), chat_histories, interview_step, resume_summary, job_summary, "⚠️ Error: JD too short.", "⚠️ Error: JD too short.", None, "", None)
            
        try:
            resume_text = extract_text_from_pdf(resume_path)
            if not resume_text.strip(): raise ValueError("PDF is empty or unreadable.")
            
            r_summary = Resume_Analyst(resume_text)
            j_summary = Job_Description_Expert(job_desc)
            
            if "INVALID" in str(r_summary).upper():
                return (None, gr.update(), "Invalid Resume.", None, None, "", chat_histories, interview_step, None, None, "", "### ⚠️ Invalid Resume PDF.", None, "", None)
            if "INVALID" in str(j_summary).upper():
                return (None, gr.update(), "Invalid JD.", None, None, "", chat_histories, interview_step, None, None, "", "### ⚠️ Invalid Job Description.", None, "", None)

            # Randomly shuffle and select a fresh HR Persona for each new interview!
            hr_persona = random.choice(HR_PERSONAS)
            print(f"Selected HR Persona: {hr_persona['name']} ({hr_persona['gender']}, voice: {hr_persona['voice']})")

            resume_summary, job_summary = r_summary, j_summary
            chat_histories = {}
            print("Session Success.")
        except Exception as e:
            return (None, gr.update(), f"Error: {e}", None, None, "", chat_histories, interview_step, None, None, "", f"### ❌ {e}", None, "", None)
            
    # Ensure hr_persona is populated
    if not hr_persona or not isinstance(hr_persona, dict):
        hr_persona = random.choice(HR_PERSONAS)

    # 2. Process Answer from Previous Question (if step > 0)
    if interview_step > 0 and latest_question_text:
        answer_text = ""
        # Check text input first
        if user_text_ans and str(user_text_ans).strip():
            answer_text = str(user_text_ans).strip()
        # Fallback/Combine with audio transcription if available
        if user_audio_path:
            try:
                transcribed = transcribe_audio_faster_whisper(user_audio_path)
                if transcribed.strip():
                    if answer_text:
                        answer_text = f"{answer_text} (Voice: {transcribed.strip()})"
                    else:
                        answer_text = transcribed.strip()
            except Exception as e:
                print(f"Voice transcription error: {e}")
        
        # If no transcript or text provided, supply fallback so chat_histories advances
        if not answer_text:
            answer_text = "[Candidate submitted response]"
            
        chat_histories[latest_question_text] = answer_text
        print(f"Recorded answer for Q{interview_step}: {answer_text[:60]}...")
    
    # 3. Check for Completion
    if interview_step >= num_q_val and interview_step > 0:
        gr.Info("Generating final evaluation...")
        try:
            eval_data = Evaluator(chat_histories, job_summary)
            radar, bar = create_performance_charts(eval_data['scores'], eval_data['benchmarks'])
            voice_name = hr_persona.get("voice", "en-US-ChristopherNeural")
            conclusion_audio = text_to_speech(eval_data.get('spoken_conclusion', 'Thank you.'), voice_name=voice_name)

            return (conclusion_audio, gr.update(value="✅ Complete", interactive=False), 
                    eval_data['text_evaluation'], radar, bar, eval_data.get('correction_needed', ''),
                    chat_histories, interview_step + 1, resume_summary, job_summary, "", "### 🏁 Interview Complete!", None, "", hr_persona)
        except Exception as e:
            return (None, gr.update(), f"Evaluation error: {e}", None, None, "", chat_histories, interview_step, resume_summary, job_summary, "", "### ❌ Evaluation Failed.", None, "", hr_persona)

    # 4. Generate Next Question
    try:
        print(f"Generating Question {interview_step + 1} of {num_q_val}...")
        question = Interviewer(chat_histories, resume_summary or "Candidate", job_summary or "Role", current_step=interview_step+1, total_q=num_q_val, hr_persona=hr_persona)
        
        hr_name = hr_persona.get("name", "HR Coach")
        if "Error:" in question:
            return (None, gr.update(), question, None, None, "", chat_histories, interview_step, resume_summary, job_summary, question, f"### 🧔 {hr_name}: \n⚠️ {question}", None, "", hr_persona)

        voice_name = hr_persona.get("voice", "en-US-ChristopherNeural")
        audio_file = text_to_speech(question, voice_name=voice_name)
        button_label = f"Submit Answer & Next ({interview_step + 1}/{num_q_val})"
        q_md = f"### 🧔 {hr_name}:\n{question}"
        
        return (audio_file, gr.update(value=button_label, interactive=True), 
                "Evaluation will appear at the end of the interview.", None, None, "",
                chat_histories, interview_step + 1, resume_summary, job_summary, question, q_md, None, "", hr_persona)
    except Exception as e:
        err = f"Generation error: {e}"
        return (None, gr.update(), err, None, None, "", chat_histories, interview_step, resume_summary, job_summary, err, f"### ❌ {err}", None, "", hr_persona)

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

// PWA Service Worker Registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('✅ PWA Service Worker registered with scope:', registration.scope);
            })
            .catch(function(err) {
                console.error('❌ PWA Service Worker registration failed:', err);
            });
    });
}

window.startInterviewTimer = function() {
    console.log("⏱️ Interview started...");
};

// Splash handling is now primarily driven by CSS for robustness
window.addEventListener('load', () => {
    console.log("Page fully loaded.");
    const splash = document.getElementById('splash-overlay');
    if (splash) splash.classList.add('hide-splash');
});

// Fallback for safety
setTimeout(() => {
    const splash = document.getElementById('splash-overlay');
    if (splash && !splash.classList.contains('hide-splash')) {
        splash.classList.add('hide-splash');
    }
}, 4000);

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
    try {
        const answers = {
            1: "I analyze your resume and job description to create tailored questions that simulate a real interview experience.",
            2: "I use Groq-powered LLaMA 3.3 for intelligence and Faster-Whisper for high-speed voice recognition.",
            3: "Absolutely. I process your data in real-time and never store your documents or audio on any server.",
            4: "Complete the interview (all questions) and then check the 'Analytics' tab for your detailed performance breakdown.",
            5: "For the best experience, provide a clear job description including Job Title, Key Responsibilities, and Required Skills (Technical & Tools)."
        };
        const display = document.getElementById('faq-answer-display');
        if (!display) return;
        
        display.innerText = answers[qId];
        display.style.display = 'block';
        display.style.opacity = '1';
    } catch(e) { console.error("FAQ Error:", e); }
};

setInterval(() => {
    try {
        if (!document.body.classList.contains('dark')) document.body.classList.add('dark');
        
        const hr = document.getElementById('hr-character');
        const speech = document.getElementById('speech-bubble');
        if (hr && speech && !hr.dataset.rdy) {
            hr.onmouseenter = () => { if(speech) speech.style.opacity = '1'; };
            hr.onmouseleave = () => { if(speech) speech.style.opacity = '0'; };
            hr.dataset.rdy = "true";
        }
    } catch(e) {}
}, 2000);
"""

custom_css = """
/* Non-blocking Decorative Splash */
#splash-overlay {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: #020202;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    z-index: 999999;
    opacity: 1;
    visibility: visible;
    transition: opacity 1s ease, visibility 1s ease;
    pointer-events: none;
}

#splash-overlay.hide-splash {
    opacity: 0;
    visibility: hidden;
}

#splash-logo {
    width: 280px; height: auto;
    opacity: 0;
    transform: scale(0.5);
    animation: popOut 1.2s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards, pulseGlow 2s 1.2s infinite alternate;
}
@keyframes popOut {
    0% { opacity: 0; transform: scale(0.3); }
    100% { opacity: 1; transform: scale(1); }
}
@keyframes pulseGlow { 
    from { filter: drop-shadow(0 0 5px #00d2ff); transform: scale(1); } 
    to { filter: drop-shadow(0 0 25px #3a7bd5); transform: scale(1.05); } 
}
@keyframes fadeIn { to { opacity: 1; visibility: visible; } }

#main-app-content {
    display: block !important;
    padding-top: 10px !important;
    opacity: 1 !important;
    visibility: visible !important;
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

# Dynamic PWA Manifest Generation
custom_head = f"""
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#050505">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="AI Coach">
<link rel="apple-touch-icon" href="/logo.png">
<script>
{custom_js}
</script>
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, head=custom_head) as demo:
    chat_histories_state = gr.State()
    interview_step_state = gr.State(0)
    resume_summary_state = gr.State(None)
    job_summary_state = gr.State(None)
    latest_question_text_state = gr.State("")
    hr_persona_state = gr.State(None)
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
                question_display = gr.Markdown("", elem_id="question-display")
                interviewer_question = gr.Audio(label="🧔 Interviewer Speaks:", type="filepath", interactive=False, autoplay=True)
                dummy_mic_status = gr.Textbox(visible=False, elem_id="dummy-mic-status")
                user_answer = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Your Answer (Voice)")
                user_text_answer = gr.Textbox(label="✍️ Or Type Your Answer (Text Fallback)", lines=3, placeholder="If mic is off, silent, or disabled, type your answer here...")
                
        # Separation for Evaluation and Analytics with explicit class for spacing
        with gr.Tabs(elem_classes="tabs-container") as tabs:
            with gr.Tab("📝 Detailed Evaluation"):
                gr.HTML("<div style='margin-bottom: 20px; font-weight: bold; color: #00d2ff; text-transform: uppercase; letter-spacing: 2px;'>HR Feedback & Roadmap</div>")
                evaluation_textbox = gr.Markdown("", elem_classes="evaluation-md-box")
            with gr.Tab("📊 Performance Analytics"):
                with gr.Row(elem_classes="analytics-responsive-row"):
                    radar_plot = gr.Plot(label="Skill Competency", elem_classes="analytics-plot")
                    bar_plot = gr.Plot(label="Peer Benchmarks", elem_classes="analytics-plot")
                correction_md = gr.Markdown("", elem_id="correction-needed-md")

    # 3. Interactive HR Character Overlay
    gr.HTML(f"""
        <div id="hr-fixed-wrapper">
            <div id="faq-chatbot">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <div class="chat-title" style="margin-bottom: 0;">❓ FAQs</div>
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
        inputs=[resume_input, job_desc_input, num_q_input, interviewer_question, user_answer, user_text_answer, chat_histories_state, interview_step_state, resume_summary_state, job_summary_state, latest_question_text_state, hr_persona_state],
        outputs=[interviewer_question, start_btn, evaluation_textbox, radar_plot, bar_plot, correction_md, chat_histories_state, interview_step_state, resume_summary_state, job_summary_state, latest_question_text_state, question_display, user_answer, user_text_answer, hr_persona_state]
    )

    user_answer.start_recording(
        fn=lambda: "started",
        inputs=[],
        outputs=[dummy_mic_status],
        js="function() { if(window.startInterviewTimer) { window.startInterviewTimer(); } return []; }"
    )
    
    # 6. Final cleanup (HTML script injection removed because we use 'head' arg in blocks now)

# --- FastAPI App with Full PWA Routes ---
app = FastAPI()

@app.get("/manifest.json")
def get_manifest():
    manifest_data = {
        "name": "AI Interview Coach",
        "short_name": "AICoach",
        "start_url": "/",
        "display": "standalone",
        "orientation": "portrait",
        "background_color": "#050505",
        "theme_color": "#050505",
        "description": "Your Personalized AI Interview Coach",
        "icons": [
            {
                "src": "/logo.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable"
            }
        ]
    }
    return Response(content=json.dumps(manifest_data), media_type="application/json")

@app.get("/sw.js")
def get_sw():
    sw_code = """
    const CACHE_NAME = 'ai-coach-v2';
    const urlsToCache = ['/', '/manifest.json', '/logo.png'];

    self.addEventListener('install', (event) => {
        self.skipWaiting();
        event.waitUntil(
            caches.open(CACHE_NAME).then((cache) => cache.addAll(urlsToCache).catch(() => {}))
        );
    });

    self.addEventListener('activate', (event) => {
        event.waitUntil(
            caches.keys().then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cache) => {
                        if (cache !== CACHE_NAME) {
                            return caches.delete(cache);
                        }
                    })
                );
            }).then(() => self.clients.claim())
        );
    });

    self.addEventListener('fetch', (event) => {
        if (event.request.method !== 'GET') return;
        event.respondWith(
            fetch(event.request).catch(() => caches.match(event.request))
        );
    });
    """
    return Response(content=sw_code, media_type="application/javascript")

@app.get("/logo.png")
def get_pwa_logo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, "logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    return Response(content=b"", media_type="image/png")

# Enable Queue on demo Blocks
demo.queue()

# Mount Gradio app onto FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
