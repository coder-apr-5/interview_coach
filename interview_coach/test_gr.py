import gradio as gr
def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
print("Starting minimal app...")
demo.launch(share=False)
