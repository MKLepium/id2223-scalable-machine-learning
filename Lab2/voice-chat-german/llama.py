import requests
import json
import string

url = 'https://llama.app.cloud.cbh.kth.se/completion'
headers = {}
# headers = {
#     'authority': 'llama.app.cloud.cbh.kth.se',
#     'accept': 'text/event-stream',
#     'accept-language': 'sv,en;q=0.9,en-GB;q=0.8,en-US;q=0.7',
#     'content-type': 'application/json',
#     'origin': 'https://llama.app.cloud.cbh.kth.se',
#     'referer': 'https://llama.app.cloud.cbh.kth.se/',
#     'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Microsoft Edge";v="116"',
#     'sec-ch-ua-mobile': '?0',
#     'sec-ch-ua-platform': '"macOS"',
#     'sec-fetch-dest': 'empty',
#     'sec-fetch-mode': 'cors',
#     'sec-fetch-site': 'same-origin',
#     'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.69',
# }

USERTOKEN="<|prompter|>"
ASSISTANTTOKEN="<|assistant|>"
ENDTOKEN="<|endoftext|>"
PUBLIC_PREPROMPT="Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful."
# PUBLIC_PREPROMPT="Nachfolgend finden Sie eine Reihe von Dialogen zwischen verschiedenen Personen und einem KI-Assistenten. Die KI versucht hilfsbereit, höflich, ehrlich, kultiviert, emotional bewusst und bescheiden, aber kenntnisreich zu sein. Der Assistent hilft Ihnen gerne bei fast allem und wird sein Bestes tun, um genau zu verstehen, was benötigt wird. Außerdem wird versucht, falsche oder irreführende Informationen zu vermeiden, und es gibt Vorbehalte, wenn man sich über die richtige Antwort nicht ganz sicher ist. Allerdings ist der Assistent praktisch und gibt wirklich sein Bestes und lässt die Vorsicht nicht zu sehr in die Quere kommen."

data = {
    "stream": True,
    "n_predict": 400,
    "temperature": 0.7,
    "stop": ["</s>", "llama:", "Nutzer:"],
    "repeat_last_n": 256,
    "repeat_penalty": 1.18,
    "top_k": 40,
    "top_p": 0.5,
    "tfs_z": 1,
    "typical_p": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "mirostat": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "grammar": "",
    "prompt": "",
    'ban_eos_token': False,
    'skip_special_tokens': True,
    'stopping_strings': [ENDTOKEN, f"{USERTOKEN.strip()}", f"{USERTOKEN.strip()}:", f"{ENDTOKEN}{USERTOKEN.strip()}", f"{ENDTOKEN}{USERTOKEN.strip()}:", f"{ASSISTANTTOKEN.strip()}", f"{ASSISTANTTOKEN.strip()}:", f"{ENDTOKEN}{ASSISTANTTOKEN.strip()}:", f"{ENDTOKEN}{ASSISTANTTOKEN.strip()}"],
}

printable = string.ascii_letters + string.digits + string.punctuation + ' '
def hex_escape(s):
    return ''.join(c if c in printable else r'\x{0:02x}'.format(ord(c)) for c in s)

def ask_llama(query):
    data["prompt"] = PUBLIC_PREPROMPT + query

    result = []
    with requests.Session() as session:
        # Send the initial request
        response = session.post(url, headers=headers, json=data, stream=True, verify=True)

        # Check for a successful connection
        if response.status_code == 200:
            print("Connected to the stream!")

            # Iterate over the lines of the response content
            for line in response.iter_lines(decode_unicode=False):
                if line:
                    print("Line:", line)
                    utf8_line = line.decode('utf-8')
                    line_data = json.loads(utf8_line[5:])  # Remove "data: " prefix and parse JSON
                    content = line_data.get("content")
                    stop = line_data.get("stop")
                    result.append(content)
            # print(result)

        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")
    result = "".join(result)
    print("Result: " + result)
    # conversation.append(result + "\n\n")
    return result

def ask_llama_yield(query):
    data["prompt"] = query

    with requests.Session() as session:
        # Send the initial request
        response = session.post(url, headers=headers, json=data, stream=True, verify=False)

        # Check for a successful connection
        if response.status_code == 200:
            print("Connected to the stream!")

            # Iterate over the lines of the response content
            for line in response.iter_lines(decode_unicode=False):
                if line:
                    # print(line)
                    utf8_line = line.decode('utf-8')
                    line_data = json.loads(utf8_line[5:])  # Remove "data: " prefix and parse JSON
                    print(line_data)
                    content = line_data.get("content")
                    stop = line_data.get("stop")
                    yield content
            # print(result)

        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")

if __name__ == "__main__":
    while True:
        print(ask_llama(input()))