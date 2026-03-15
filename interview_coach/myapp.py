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
        raise gr.Error("⚠️ GROQ_API_KEY is not set. Please add it to your .env file. Get a free key at https://console.groq.com/keys")

    llm_client = Groq(api_key=api_key)
    return llm_client

def chat_with_llm(system_prompt, user_prompt, max_retries=3):
    """Send a chat message to Groq (LLaMA 3.3 70B) with retry on rate limits."""
    client = get_llm()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4096,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 30 * (attempt + 1)
                print(f"⏳ Rate limited. Waiting {wait_time}s before retry ({attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise gr.Error(f"⚠️ LLM Error: {str(e)}")
    
    raise gr.Error("⚠️ Rate limit exceeded after retries. Please wait a moment and try again.")

def Resume_Analyst(resume):
    prompt = f"""
    Write a detailed REPORT, in several paragraphs, on the candidate. 
    Three paragraphs: candidate's name and demographic info, key_skills, and the summary of the past experiences.

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

    Answer Histories:
    {chat_histories}

    Resume Summary:
    {resume_summary}

    Job Summary:
    {job_summary}
    """
    return chat_with_llm("You are job expert.", prompt)

def Interviewer(resume_summary, job_summary, action=None, last=False):
    if not last:
        if action is not None:
            prompt = f"""
            Directly ask the question based on the given action instruction, 
            resume summary and the job summary.

            DO NOT GIVE ANY EXPLANATIONS WHY YOU ASK THE QUESTION.

            Action:
            {action}

            Resume Summary:
            {resume_summary}

            Job Summary:
            {job_summary}
            """
            return chat_with_llm("You are an expert interviewer.", prompt)
        else:
            return "Tell me about yourself."
    else:
        prompt = f"""
            The interview ends. Wrap up and express gratitude towards the candidate based on the resume.
            Be CONCISE.

            Resume Summary:
            {resume_summary}
            """
        return chat_with_llm("You are an expert interviewer.", prompt)

def Evaluator(chat_histories, job_summary):
    prompt = f"""
    Based on the histories of the answers and the job summary,
    evaluate if the candidate is a good match, by personality and skills, 
    and give reasons.

    Answer Histories:
    {chat_histories}

    Job Summary:
    {job_summary}
    """
    return chat_with_llm("You are an expert to judge the performance of an interviewee.", prompt)

# Function to read the pdf file (for resume)
def extract_text_from_pdf(pdf_file_path):
    file_path_str = pdf_file_path if isinstance(pdf_file_path, str) else pdf_file_path.name
    reader = PyPDF2.PdfReader(file_path_str)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# Function to generate the audio file
def text_to_speech_file(text_input):
    audio_file_path = "temp_voice.mp3"
    tts = gTTS(text=text_input, lang='en')
    tts.save(audio_file_path)
    return audio_file_path

# Function to convert the audio file to text
def transcribe_audio_faster_whisper(
    audio_file_path: str, 
    model_size: str = "base", 
    device: str = "auto",
    compute_type: str = "auto"
) -> str:
    
    if audio_file_path is None:
        return "Please provide an audio input."
        
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
        
    device = "cpu"
        
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(audio_file_path, beam_size=5)
        full_transcript = [segment.text for segment in segments]
        return "".join(full_transcript).strip()

    except Exception as e:
        return f"❌ An error occurred during transcription: {e}"

def next_question(resume_path, job_str, total_number, question_previous, answer_previous, chat_histories, interview_step, resume_summary, job_summary, latest_question_text):
    chat_histories = chat_histories or {}
    interview_step = interview_step or 0
    latest_question_text = latest_question_text or ""
    
    # Validate inputs
    if resume_path is None:
        raise gr.Error("⚠️ Please upload your resume (PDF) before starting the interview.")
    if not job_str or job_str.strip() == "":
        raise gr.Error("⚠️ Please paste the job description before starting the interview.")

    # Generate resume_summary if it hasn't been done
    if resume_summary is None:
        resume_summary = extract_text_from_pdf(resume_path)
        
    # Generate job_summary if it hasn't been done
    if job_summary is None:
        job_summary = Job_Description_Expert(job_str)
    
    # Transcribe user's audio answer
    try:
        answer_previous = transcribe_audio_faster_whisper(answer_previous)
    except:
        answer_previous = ""
    
    # Update chat history
    if interview_step > 0:
        chat_histories[f"Q{interview_step}: {latest_question_text}"] = answer_previous
    
    # Generate next question
    if interview_step < total_number:
        if interview_step == 0:
            action = None
        else:
            chat_hist_str = str(chat_histories)
            action = Interview_Question_Action(chat_hist_str, resume_summary, job_summary)
        
        Question_next = Interviewer(resume_summary, job_summary, action, last=False)
    else:
        Question_next = Interviewer(resume_summary, job_summary, action=None, last=True)
    
    # Evaluate if interview ends
    if interview_step >= total_number:
        evaluation = Evaluator(str(chat_histories), job_summary)
        chat_histories = {}
        interview_step = 0
        resume_summary = None
        job_summary = None
    else:
        evaluation = "Evaluation Ongoing ......"
    
    # Convert question to audio
    question_audio_path = text_to_speech_file(Question_next)
    latest_question_text = Question_next
    interview_step += 1

    return gr.update(value=question_audio_path), gr.update(value=None), gr.update(value="Next Question"), gr.update(value=evaluation), chat_histories, interview_step, resume_summary, job_summary, latest_question_text

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    chat_histories_state = gr.State({})
    interview_step_state = gr.State(0)
    resume_summary_state = gr.State(None)
    job_summary_state = gr.State(None)
    latest_question_text_state = gr.State("")

    gr.Markdown("# 🎤 Personalized Interview Coach")
    gr.Markdown('## Upload your pdf resume/CV and copy paste the job description you are applying to:')
    
    with gr.Row():
        resume_input = gr.File(label="Upload Resume (PDF)", type='filepath')
        job_desc_input = gr.Textbox(label="Job Description", lines=15)
        
    gr.Markdown('## Decide the length of your mock interview (from 1 question to 10 questions):')
    num_q_input = gr.Slider(label="Number of Questions", minimum=1, maximum=10, value=5, step=1)
    
    gr.Markdown('## Click "Start Interview" below to start the Mock Interview!')
    interviewer_question = gr.Audio(label="Interviewer Question", type="filepath")
    
    user_answer = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Your turn! Record Your Answer."
        )
        
    start_btn = gr.Button("🚀 Start Interview", scale=2, min_width=200)
    
    gr.Markdown("## Evaluating your performance along the way ...")
    evaluation_textbox = gr.Textbox(label="Performance Evaluation", lines=20)
    
    start_btn.click(
        fn=next_question,
        inputs=[resume_input, job_desc_input, num_q_input, interviewer_question, user_answer, chat_histories_state, interview_step_state, resume_summary_state, job_summary_state, latest_question_text_state],
        outputs=[interviewer_question, user_answer, start_btn, evaluation_textbox, chat_histories_state, interview_step_state, resume_summary_state, job_summary_state, latest_question_text_state]
    )

if __name__ == "__main__":
    demo.launch(share=True)
