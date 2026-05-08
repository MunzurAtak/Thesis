import json
import urllib.error
import urllib.request

from src.llms.base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    """
    LLM wrapper for locally running models through Ollama.

    Ollama must be installed and running locally.
    Example model names:
    - qwen2.5:3b
    - llama3.1:8b
    - gemma2:2b
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        stance: str,
        topic: str,
        round_number: int,
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        response = self._post_json("/api/generate", payload)

        generated_text = response.get("response", "").strip()

        if not generated_text:
            raise RuntimeError(
                f"Ollama returned an empty response for model: {self.model_name}"
            )

        return generated_text

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}{endpoint}"

        request_data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)

        except urllib.error.URLError as error:
            raise ConnectionError(
                "Could not connect to Ollama. "
                "Make sure Ollama is installed and running, then try again. "
                f"URL: {url}. Original error: {error}"
            ) from error

    def metadata(self) -> dict:
        return {
            "backend": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
