import os
import pytest
from unittest.mock import patch
from schemas.config import ProviderConfig
from providers import create_llm_provider, create_image_provider, create_audio_provider


def test_create_openai_llm_provider():
    config = ProviderConfig.model_validate({
        "type": "openai", "model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"
    })
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        provider = create_llm_provider(config)
    assert provider.model == "gpt-4o"


def test_create_anthropic_llm_provider():
    config = ProviderConfig.model_validate({
        "type": "anthropic", "model": "claude-sonnet-4-20250514", "api_key_env": "ANTHROPIC_API_KEY"
    })
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        provider = create_llm_provider(config)
    assert provider.model == "claude-sonnet-4-20250514"


def test_create_openai_image_provider():
    config = ProviderConfig.model_validate({
        "type": "openai", "model": "dall-e-3", "api_key_env": "OPENAI_API_KEY"
    })
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        provider = create_image_provider(config)
    assert provider.model == "dall-e-3"


def test_create_openai_audio_provider():
    config = ProviderConfig.model_validate({
        "type": "openai", "model": "tts-1-hd", "api_key_env": "OPENAI_API_KEY"
    })
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        provider = create_audio_provider(config)
    assert provider.model == "tts-1-hd"


def test_unknown_provider_type_raises():
    config = ProviderConfig.model_validate({
        "type": "unknown", "model": "x", "api_key_env": "KEY"
    })
    with patch.dict(os.environ, {"KEY": "test"}):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider(config)
