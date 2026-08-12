# llm_providers.py
from abc import ABC, abstractmethod
from logger import get_logger
from config import OPENAI_API_KEY

logger = get_logger("llm_providers")


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import ollama
        logger.debug(f"OllamaProvider.generate — modèle : {self.model}")
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY non définie dans .env")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        logger.debug(f"OpenAIProvider.generate — modèle : {self.model}")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


def get_provider(provider_name: str, model: str) -> LLMProvider:
    logger.info(f"Initialisation provider : {provider_name} / {model}")
    if provider_name == "ollama":
        return OllamaProvider(model=model)
    if provider_name == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("Clé API OpenAI non configurée dans .env")
        return OpenAIProvider(model=model)
    raise ValueError(f"Provider inconnu : {provider_name}")