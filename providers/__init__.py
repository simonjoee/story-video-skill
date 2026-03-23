import os
from schemas.config import ProviderConfig
from providers.base import LLMProvider, ImageProvider, AudioProvider


def create_llm_provider(config: ProviderConfig) -> LLMProvider:
    api_key = os.environ[config.api_key_env]
    match config.type:
        case "openai":
            from providers.llm.openai_provider import OpenAILLMProvider
            return OpenAILLMProvider(model=config.model, api_key=api_key)
        case "anthropic":
            from providers.llm.anthropic_provider import AnthropicLLMProvider
            return AnthropicLLMProvider(model=config.model, api_key=api_key)
        case _:
            raise ValueError(f"Unknown LLM provider: {config.type}")


def create_image_provider(config: ProviderConfig) -> ImageProvider:
    api_key = os.environ[config.api_key_env]
    match config.type:
        case "openai":
            from providers.image.openai_provider import OpenAIImageProvider
            return OpenAIImageProvider(model=config.model, api_key=api_key)
        case _:
            raise ValueError(f"Unknown image provider: {config.type}")


def create_audio_provider(config: ProviderConfig) -> AudioProvider:
    api_key = os.environ[config.api_key_env]
    match config.type:
        case "openai":
            from providers.audio.openai_provider import OpenAIAudioProvider
            return OpenAIAudioProvider(model=config.model, api_key=api_key)
        case _:
            raise ValueError(f"Unknown audio provider: {config.type}")
