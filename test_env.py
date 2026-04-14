
import os
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

def test_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is missing")
        return
    client = Groq(api_key=api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.3-70b-versatile",
        )
        print("Groq OK:", chat_completion.choices[0].message.content)
    except Exception as e:
        print("Groq Error:", e)

def test_gtts():
    try:
        tts = gTTS(text="Hello", lang='en')
        tts.save("test.mp3")
        print("gTTS OK")
        if os.path.exists("test.mp3"):
            os.remove("test.mp3")
    except Exception as e:
        print("gTTS Error:", e)

if __name__ == "__main__":
    test_groq()
    test_gtts()
