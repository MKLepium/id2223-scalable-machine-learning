import gradio as gr
import numpy as np

from gtts import gTTS
from transformers import pipeline
from llama import ask_llama
from llama import ask_llama_yield

from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="German", task="transcribe", is_multilingual=False)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="German", task="transcribe", is_multilingual=False)
custom_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
# custom_model = model_loader().model
transcriber = pipeline("automatic-speech-recognition", model=custom_model, tokenizer=tokenizer, feature_extractor=processor.feature_extractor)
SPEECH_FILE = 'speech.mp3'


def text_to_speech(text):
    speech = gTTS(text=text, lang='de', slow=False) 
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

def user(user_message, history):
    return "", history + [[user_message, None]]

def create_query(history):
    #query = "This is a conversation between user and llama, a friendly chatbot. respond in simple text. NOT MARKDOWN.\n\n"
    query = "Dies ist eine Konversation zwischen einem Nutzer und llama, einem freundlichen chatbot. antworte in einfachem text. Antworte in deutsch.\nUser: hallo 😍\nllama: Hallo, wie kann ich Ihnen heute helfen?\n"
    for message in history:
        query += "Nutzer: " + message[0] + "\nllama: " + (message[1] + "\n" if message[1] else "")
    print("query: ", query)
    return query

def bot(history):
    print("bot")
    print("history", history)
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
        height=800,
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