import os
import gradio as gr
from demo import Perception, Reasoning

information = ''

def pil_to_img_file(image):
    image.save('temp.png')
    return None

def merge_prompt(prompt_version, reasoning_module):
    if prompt_version == 'query-specific':
        return f'query-specific_{reasoning_module}'
    else:
        return prompt_version

def perception(perception_module, reasoning_module, prompt_version, text, image):
    prompt_version = merge_prompt(prompt_version, reasoning_module)
    pil_to_img_file(image)
    perception_module = Perception(prompt_version, perception_module)
    prompt = perception_module.fetch_prompt(text)
    res = perception_module.generate(prompt, 'temp.png')
    global information
    information = res
    return [(prompt, res)]

def reasoning(reasoning_module, question):
    reasoning_module = Reasoning(reasoning_module)
    res = reasoning_module.generate(information, question)
    return [(question,res)]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                perception_module = gr.Dropdown(
                    ['GPT4V', 'llava_next_yi_34b'],
                    label = "perception module"
                )
                reasoning_module = gr.Dropdown(
                    ['chatgpt-0125', 'gpt-4-0125', 'llama3-70b-chat', 'deepseek-chat'],
                    label = "reasoning module"
                )
            prompt_version = gr.Dropdown(
                ['generic', 'query-specific'],
                label = "prompt_version"
            )
            msg = gr.Textbox()
            img = gr.Image(label="Upload Image", type="pil")
            examples = gr.Examples(
                [
                    ['assets/case1.png', 'What is the relative position of the man and the woman sitting at the table?'],
                    ['assets/case2.png', 'Which number is missing?'],
                    ['assets/case3.png', 'For case A accompanying table, answer the questions that follow. Calculate the future value of the annuity, assuming that it is an ordinary annuity.']
                ],
                [img, msg]
            )
        with gr.Column(scale=2):
            perception_chatbot = gr.Chatbot(label="Perception results")
            reasoning_chatbot = gr.Chatbot(label="Reasoning results")
            with gr.Row():
                perception_but = gr.Button("Start Perception")
                reasoning_but = gr.Button("Start Reasoning")
                clear_but = gr.ClearButton([msg, perception_chatbot, reasoning_chatbot])

    perception_but.click(perception, [perception_module, reasoning_module, prompt_version, msg, img], perception_chatbot, queue=False)
    reasoning_but.click(reasoning, [reasoning_module, msg], reasoning_chatbot, queue=True)

    perception_stage = msg.submit(perception, [perception_module, reasoning_module, prompt_version, msg, img], perception_chatbot, queue=False)
    reasoning_stage = perception_stage.then(reasoning, [reasoning_module, msg], reasoning_chatbot, queue=True)

demo.launch() 