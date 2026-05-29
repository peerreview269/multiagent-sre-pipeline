"""
LocalLLM — HTTP client for a llama.cpp server.

Communicates with a locally-served LLM via llama.cpp's OpenAI-compatible
HTTP API. Requires llama.cpp's `llama-server` binary to be running on
localhost:8080 (or another endpoint passed via server_url).

Start the server before running the pipeline:

    ./llama.cpp/llama-server \\
        --model ./models/gemma-4-31B-it-Q5_K_M.gguf \\
        --n-gpu-layers -1 \\
        --port 8080

The pipeline does not require llama-cpp-python — it talks to the server
over plain HTTP, so any GGUF model served via llama-server works.
"""

import requests


class LocalLLM:
    def __init__(self, model_path: str, n_gpu_layers: int = -1,
                 n_ctx: int = 16384, temperature: float = 0.7,
                 server_url: str = "http://127.0.0.1:8080"):
        self.server_url = server_url.rstrip("/")
        self.model_path = model_path
        # Verify server is reachable
        try:
            resp = requests.get(f"{self.server_url}/v1/models", timeout=10)
            resp.raise_for_status()
            models = resp.json()
            model_name = models["data"][0]["id"] if models.get("data") else "unknown"
            print(f"  Connected to llama.cpp server: {model_name}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach llama.cpp server at {self.server_url}. "
                f"Start it with:\n"
                f"  ./llama.cpp/llama-server --model {model_path} "
                f"--n-gpu-layers -1 --port 8080\n"
                f"Error: {e}"
            )

    def _chat(self, prompt: str, max_tokens: int, temperature: float,
              top_p: float, top_k: int, stop: list[str]) -> str:
        """
        Call the llama.cpp server /v1/completions endpoint with a raw prompt.
        We use completions (not chat/completions) because we build the prompt
        manually in build_chat() to control the exact token format.
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop,
            "stream": False,
            "cache_prompt": True,    # reuse KV cache across calls
        }
        try:
            resp = requests.post(
                f"{self.server_url}/v1/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"].strip()
        except requests.exceptions.Timeout:
            return ""
        except Exception as e:
            print(f"\n  [!] Server error: {e}")
            return ""

    def call(self, prompt: str, max_tokens: int = 6000,
             stop: list[str] | None = None) -> str:
        """Non-thinking call — fast structured output for specialist agents."""
        stop = stop or ["<turn|>"]
        return self._chat(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            stop=stop,
        )

    def call_thinking(self, prompt: str, max_tokens: int = 6000,
                      stop: list[str] | None = None) -> str:
        """Thinking call — full reasoning for supervisor, TieAgent, CriticAgent, ER."""
        stop = stop or ["OBSERVATION:", "<turn|>"]
        return self._chat(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            stop=stop,
        )
