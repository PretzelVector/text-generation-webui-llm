from typing import List, Optional

import aiohttp
import requests
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM


class TextGenerationWebUILLM(LLM):
    """A Simple wrapper around https://github.com/oobabooga/text-generation-webui API for use with langchain."""

    api_host: str = "http://localhost:5000"
    generate_endpoint: str = "/api/v1/generate"
    token_count_endpoint: str = "/api/v1/token-count"

    max_new_tokens: int = 250
    do_sample: bool = True
    temperature: float = 1.3
    top_p: float = 0.1
    typical_p: float = 1
    epsilon_cutoff: float = 0  # In units of 1e-4,
    eta_cutoff: float = 0  # In units of 1e-4,
    repetition_penalty: float = 1.18
    top_k: int = 40
    min_length: int = 0
    no_repeat_ngram_size: int = 0
    num_beams: int = 1
    penalty_alpha: float = 0
    length_penalty: float = 1
    early_stopping: bool = False
    mirostat_mode: int = 0
    mirostat_tau: int = 5
    mirostat_eta: float = 0.1
    seed: int = -1
    add_bos_token: bool = True
    truncation_length: int = 2048
    ban_eos_token: bool = False
    skip_special_tokens: bool = True
    stopping_strings: List[str] = []

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "text-generation-webui"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        request = self._create_request(prompt, stop)
        api_endpoint = f"{self.api_host}{self.generate_endpoint}"
        response = requests.post(api_endpoint, json=request)
        response.raise_for_status()

        return response.json()["results"][0]["text"]

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> str:
        request = self._create_request(prompt, stop)
        api_endpoint = f"{self.api_host}{self.generate_endpoint}"

        async with aiohttp.ClientSession() as session:
            async with session.post(api_endpoint, json=request) as response:
                response.raise_for_status()

                return (await response.json())["results"][0]["text"]

    def _create_request(self, prompt: str, stop: Optional[List[str]] = None) -> dict:
        return {
            "prompt": prompt,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "epsilon_cutoff": self.epsilon_cutoff,
            "eta_cutoff": self.eta_cutoff,
            "repetition_penalty": self.repetition_penalty,
            "top_k": self.top_k,
            "min_length": self.min_length,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "num_beams": self.num_beams,
            "penalty_alpha": self.penalty_alpha,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "mirostat_mode": self.mirostat_mode,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
            "seed": self.seed,
            "add_bos_token": self.add_bos_token,
            "truncation_length": self.truncation_length,
            "ban_eos_token": self.ban_eos_token,
            "skip_special_tokens": self.skip_special_tokens,
            "stopping_strings": self.stopping_strings + (stop or []),
        }

    def get_num_tokens(self, prompt: str) -> int:
        api_endpoint = f"{self.api_host}{self.token_count_endpoint}"
        response = requests.post(api_endpoint, json={"prompt": prompt})
        response.raise_for_status()

        return response.json()["results"][0]["tokens"]


