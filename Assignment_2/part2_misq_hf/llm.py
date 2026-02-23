import requests
import time

class LLM:
    def __init__(self, model="llama3.1:8b", url="http://localhost:11434"):
        self.model = model
        self.url = url
        self.max_retries = 3
        self.timeout = 30

    def _post_with_retry(self, endpoint, payload, call_name):
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(
                    f"{self.url}{endpoint}",
                    json=payload,
                    timeout=self.timeout
                )
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError, KeyError) as err:
                last_err = err
                if attempt < self.max_retries:
                    print(f"{call_name} failed (attempt {attempt}/{self.max_retries}), retrying...")
                    time.sleep(2 ** (attempt - 1))
                else:
                    raise RuntimeError(
                        f"{call_name} failed after {self.max_retries} retries: {err}"
                    ) from err

    def chat(self, messages, temp=0):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temp}
        }
        data = self._post_with_retry("/api/chat", payload, "chat")
        try:
            return data["message"]["content"]
        except KeyError as err:
            raise RuntimeError(f"Unexpected chat response format: {data}") from err

    def generate(self, prompt, temp=0):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temp}
        }
        data = self._post_with_retry("/api/generate", payload, "generate")
        try:
            return data["response"]
        except KeyError as err:
            raise RuntimeError(f"Unexpected generate response format: {data}") from err


if __name__ == "__main__":
    llm = LLM("llama3.1:8b")
    resp = llm.chat([{"role": "user", "content": "Say hello in one sentence."}])
    print(f"Response: {resp}")

    # resp2 = llm.generate("What is 1+1?")
    # print(resp2)
