import requests
import time


class LLM:
    def __init__(self, model="llama3.1:8b", url="http://localhost:11434"):
        self.model = model
        self.url = url

    def chat(self, messages, temp=0):
        # call ollama chat interface
        try:
            r = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temp}
                }
            )
            r.raise_for_status()
            rsp = r.json()["message"]["content"]
            # print(rsp)
            return rsp
        except:
            print("chat failed, retrying...")
            time.sleep(2)
            return self.chat(messages, temp)

    def generate(self, prompt, temp=0):
        # single-turn generate, no history
        try:
            r = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temp}
                }
            )
            r.raise_for_status()
            return r.json()["response"]
        except:
            print("generate failed, retrying...")
            time.sleep(2)
            return self.generate(prompt, temp)


if __name__ == "__main__":
    llm = LLM("llama3.1:8b")
    resp = llm.chat([{"role": "user", "content": "Say hello in one sentence."}])
    print(f"Response: {resp}")

    # resp2 = llm.generate("What is 1+1?")
    # print(resp2)
