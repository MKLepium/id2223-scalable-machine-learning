from transformers import pipeline

class GPT4All:
    def __init__(self):
        self.generator = pipeline('text-generation', model='nomic-ai/gpt4all-13b-snoozy')

    def query(self, message):
        response = self.generator(message, max_length=50, num_return_sequences=1)
        return response[0]['generated_text']

# Example usage
if __name__ == "__main__":
    gpt = GPT4All()
    message = "Hello, how are you?"
    response = gpt.query(message)
    print(response)