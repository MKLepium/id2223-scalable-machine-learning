import gradio as gr
import random
import numpy as np
import time

from gtts import gTTS
from transformers import pipeline
from llama import ask_llama
from llama import ask_llama_yield

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-medium")
SPEECH_FILE = 'speech.mp3'

def text_to_speech(text):
    speech = gTTS(text=text, lang='en', slow=False) 
    speech.save(SPEECH_FILE) 
    return SPEECH_FILE

def user(user_message, history):
    return "", history + [[user_message, None]]

def asr(audio: np.ndarray):
    print("asr")
    if audio is None:
        return ""
    sr, y = audio
    y = np.copy(y)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def gpt(query, chatbot: gr.Chatbot):
    print(chatbot)
    response = ask_llama(query)
    print("response: " + response)  
    return [(query), (response)]

def user(user_message, history):
    return "", history + [[user_message, None]]

def create_query(history):
    query = "This is a conversation between user and llama, a friendly chatbot. respond in simple text. NOT MARKDOWN.\n\n"
    for message in history:
        query += "User: " + message[0] + "\n\nllama: " + (message[1] + "\n\n" if message[1] else "")
    print("query: ", query)
    return query

def bot(history):
    print("bot")
    history[-1][1] = ask_llama(create_query(history))
    return history

    # stream the llama response, not useful with text to speech
    # for content in ask_llama_yield(create_query(history)):
    #     history[-1][1] += content
    #     time.sleep(0.05)
    #     yield history

def get_audio(history):
    last_chatbot_message = history[-1][1]
    if not last_chatbot_message:
        return None
    return text_to_speech(last_chatbot_message)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        render_markdown=True,
        elem_id="chatbot",
    )
    response_audio = gr.Audio(
        sources=['upload'],
        autoplay=True,
        visible=False,
    )
    chatbot.change(get_audio, chatbot, response_audio)

    with gr.Row(elem_id="output-container"):
        audio = gr.Audio(sources=['microphone'])
        msg = gr.Textbox(lines=2, scale=3)
        sub = gr.Button('Submit', scale=1)
    # clear = gr.Button('Clear', scale=1)

    audio.change(asr, audio, msg)

    sub.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    ).then(
        lambda: audio.clear(), queue=False
    )
    
    # clear.click(lambda: None, None, None, None, None, queue=False)
    
demo.queue()
if __name__ == "__main__":
    demo.launch(debug=True)