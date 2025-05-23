from gpt4all import GPT4All

class LocalLLM:
    def __init__(self, model_name="mistral-7b-instruct-v0.1.Q4_0.gguf"):
        self.model = GPT4All(model_name)

    def ask(self, prompt, max_tokens=200):
        # La funzione generate di GPT4All accetta temperature come argomento opzionale
        response = self.model.generate(prompt, max_tokens=max_tokens)
        
        # Pulizia base della risposta: rimuove il prompt se presente
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
