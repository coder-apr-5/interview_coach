import plotly.graph_objects as go
import json
import re

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
    """Lazily initialize the Groq client on first use."""
    global llm_client
    if llm_client is not None:
        return llm_client

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise gr.Error("⚠️ GROQ_API_KEY is not set. Please add it to your .env file.")

    llm_client = Groq(api_key=api_key)
    return llm_client

def chat_with_llm(system_prompt, user_prompt, max_retries=3, json_mode=False):
    """Send a chat message to Groq (LLaMA 3.3 70B) with retry on rate limits."""
    client = get_llm()
    
    for attempt in range(max_retries):
        try:
            params = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 4096,
                "temperature": 0.7,
            }
            if json_mode:
                params["response_format"] = {"type": "json_object"}
                
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 30 * (attempt + 1)
                print(f"⏳ Rate limited. Waiting {wait_time}s before retry ({attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise gr.Error(f"⚠️ LLM Error: {str(e)}")
    
    raise gr.Error("⚠️ Rate limit exceeded after retries.")

def Resume_Analyst(resume):
    prompt = f"""
    Write a detailed REPORT, in several paragraphs, on the candidate. 
    Resume:
    {resume}
    """
    return chat_with_llm("You are an HR expert in reviewing resumes.", prompt)

def Job_Description_Expert(job_description):
    prompt = f"""
    Write a summary of the job description.
    Identify the skills required and the experiences preferred.
    Job Description:
    {job_description}
    """
    return chat_with_llm("You are job expert.", prompt)

def Interview_Question_Action(chat_histories, resume_summary, job_summary):
    prompt = f"""
    Based on the histories of the answers, the resume summary and the job summary,
    pick one of the following two actions for the next question:
    - (1) Ask about another past experience or skills on the resume.
    - (2) Ask follow-up questions of the current topic.
    Answer Histories: {chat_histories}
    Resume Summary: {resume_summary}
    Job Summary: {job_summary}
    """
    return chat_with_llm("You are job expert.", prompt)

def Interviewer(resume_summary, job_summary, action=None, last=False):
    if not last:
        if action is not None:
            prompt = f"Directly ask the question based on Action: {action}\nResume: {resume_summary}\nJob: {job_summary}"
            return chat_with_llm("Interviewer. Ask only the question. No fluff.", prompt)
        else:
            return "Tell me about yourself."
    else:
        prompt = f"End the interview. Resume: {resume_summary}"
        return chat_with_llm("Interviewer. Wrap up concisely.", prompt)

def Evaluator(chat_histories, job_summary):
    prompt = f"""
    Evaluate the candidate performance based on history and job requirements.
    Return a JSON object with:
    1. "text_evaluation": Detailed feedback, strengths, and solutions for weaknesses.
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
            fill='none',
            name='Industry Benchmark',
            line_color='#ff0055',
            line_dash='dash'
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#444"),
            angularaxis=dict(gridcolor="#444")
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        margin=dict(l=40, r=40, t=20, b=20)
    )

    # Bar Chart for Comparison
    fig_bar = go.Figure(data=[
        go.Bar(name='You', x=categories, y=values, marker_color='#00d1ff'),
        go.Bar(name='Average', x=categories, y=[benchmarks.get(c, 50) for c in categories] if benchmarks else [50]*len(categories), marker_color='#555')
    ])
    fig_bar.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        yaxis=dict(range=[0, 100], gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
        margin=dict(l=40, r=40, t=20, b=20)
    )

    return fig_radar, fig_bar

# --- Lazy Whisper initialization ---
whisper_model = None
def get_whisper():
    global whisper_model
    if whisper_model is not None: return whisper_model
    print("⏳ Loading Whisper 'tiny' model...")
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return whisper_model

def transcribe_audio_faster_whisper(audio_file_path: str) -> str:
    if not audio_file_path: return ""
    try:
        model = get_whisper()
        segments, _ = model.transcribe(audio_file_path, beam_size=5)
        return "".join([s.text for s in segments]).strip()
    except Exception as e:
        return f"❌ Error: {e}"

def text_to_speech_file(text_input):
    audio_file_path = "temp_voice.mp3"
    gTTS(text=text_input, lang='en').save(audio_file_path)
    return audio_file_path

def extract_text_from_pdf(pdf_file_path):
    path = pdf_file_path if isinstance(pdf_file_path, str) else pdf_file_path.name
    text = "".join([page.extract_text() or "" for page in PyPDF2.PdfReader(path).pages])
    return text.strip()

def next_question(resume_path, job_str, total_number, question_previous, answer_previous, chat_histories, interview_step, resume_summary, job_summary, latest_question_text):
    chat_histories = chat_histories or {}
    interview_step = interview_step or 0
    latest_question_text = latest_question_text or ""
    
    if not resume_path: raise gr.Error("⚠️ Upload Resume.")
    if not job_str: raise gr.Error("⚠️ Paste Job Description.")

    if resume_summary is None: resume_summary = extract_text_from_pdf(resume_path)
    if job_summary is None: job_summary = Job_Description_Expert(job_str)
    
    answer_text = transcribe_audio_faster_whisper(answer_previous) if answer_previous else ""
    if interview_step > 0: chat_histories[f"Q{interview_step}: {latest_question_text}"] = answer_text
    
    eval_text = "Evaluation Ongoing ......"
    chart_radar = None
    chart_bar = None

    if interview_step < total_number:
        action = Interview_Question_Action(str(chat_histories), resume_summary, job_summary) if interview_step > 0 else None
        Question_next = Interviewer(resume_summary, job_summary, action, last=False)
    else:
        Question_next = Interviewer(resume_summary, job_summary, last=True)
        eval_data = Evaluator(str(chat_histories), job_summary)
        eval_text = eval_data.get("text_evaluation", "")
        chart_radar, chart_bar = create_performance_charts(eval_data.get("scores", {}), eval_data.get("benchmarks", {}))
        # Reset for next run
        interview_step, chat_histories, resume_summary, job_summary = -1, {}, None, None

    question_audio_path = text_to_speech_file(Question_next)
    latest_question_text = Question_next
    interview_step += 1

    return (
        gr.update(value=question_audio_path), 
        gr.update(value=None), 
        gr.update(value="Next Question" if interview_step <= total_number else "Interview Finished"), 
        gr.update(value=eval_text), 
        chart_radar, 
        chart_bar,
        chat_histories, 
        interview_step, 
        resume_summary, 
        job_summary, 
        latest_question_text
    )

custom_css = """
#hr-character {
    position: fixed;
    bottom: -10px;
    right: 20px;
    width: 220px;
    z-index: 1000;
    transition: all 0.5s ease;
    filter: drop-shadow(0 0 15px rgba(0,210,255,0.3));
    pointer-events: none;
}
#hr-character img {
    width: 100%;
    height: auto;
}
.gradio-container {
    background-color: #0d0d0d !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif;
}
.header-logo {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 20px;
}
.header-logo img {
    height: 60px;
    width: auto;
}
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    chat_histories_state = gr.State({})
    interview_step_state = gr.State(0)
    resume_summary_state = gr.State(None)
    job_summary_state = gr.State(None)
    latest_question_text_state = gr.State("")

    # Header with Logo
    gr.HTML("""
        <div class="header-logo">
            <img src="file/logo.png" alt="Logo">
            <h1 class="main-title">AI Interview Coach</h1>
        </div>
    """)
    
    gr.Markdown("### Professional Mock Interviewer Powered by LLaMA 3.3")
    
    with gr.Row():
        with gr.Column():
            resume_input = gr.File(label="📄 Resume (PDF)", type='filepath')
            job_desc_input = gr.Textbox(label="💼 Job Description", lines=10, placeholder="Paste the job requirements here...")
            num_q_input = gr.Slider(label="❓ Number of Questions", minimum=1, maximum=10, value=5, step=1)
            start_btn = gr.Button("🚀 Start Interview", variant="primary", scale=2)
        
        with gr.Column():
            interviewer_question = gr.Audio(label="🧔 Interviewer Question (Listen)", type="filepath", interactive=False)
            user_answer = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Your Answer (Record)")
            
    with gr.Tabs() as tabs:
        with gr.Tab("📝 Evaluation & Corrections"):
            evaluation_textbox = gr.Textbox(label="HR Feedback & Detailed Correction Plan", lines=15)
        with gr.Tab("📊 Analytics & Benchmarks"):
            with gr.Row():
                radar_plot = gr.Plot(label="Performance Overview")
                bar_plot = gr.Plot(label="Benchmark Comparison")

    # HR Character Overlay
    gr.HTML(f"""
        <div id="hr-character">
            <img src="file/hr_guy.png" alt="Serious HR Guy">
        </div>
    """)

    start_btn.click(
        fn=next_question,
        inputs=[resume_input, job_desc_input, num_q_input, interviewer_question, user_answer, chat_histories_state, interview_step_state, resume_summary_state, job_summary_state, latest_question_text_state],
        outputs=[interviewer_question, user_answer, start_btn, evaluation_textbox, radar_plot, bar_plot, chat_histories_state, interview_step_state, resume_summary_state, job_summary_state, latest_question_text_state]
    )

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=["e:/Interview_Coach/interview_coach/"])

