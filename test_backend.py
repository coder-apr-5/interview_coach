import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import next_question, get_llm

try:
    print("Testing get_llm()...")
    get_llm()
except Exception as e:
    print(f"Exception: {e}")

# Call next_question directly with mock data
print("Testing next_question()...")
try:
    with open("example_resume.pdf", "w") as f:
        f.write("mock pdf")
    res = next_question("example_resume.pdf", "Senior Python Developer role with Gradio experience.", 5, None, None, {}, 0, None, None, "")
    print(f"next_question generated output tuple of length: {len(res)}")
    print(f"Audio element: {res[0]}")
    print(f"Button element: {res[1]}")
    print(f"Eval block: {res[2]}")
    print(f"Question block: {res[-1]}")
except Exception as e:
    import traceback
    traceback.print_exc()
