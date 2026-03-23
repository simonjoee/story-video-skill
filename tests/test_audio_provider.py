import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from providers.audio.openai_provider import OpenAIAudioProvider


def test_openai_audio_provider_synthesize():
    fake_audio = b"\xff\xfb\x90\x00"

    mock_response = MagicMock()
    mock_response.content = fake_audio

    with patch("providers.audio.openai_provider.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.audio.speech.create = AsyncMock(return_value=mock_response)

        provider = OpenAIAudioProvider(model="tts-1-hd", api_key="test-key")
        result = asyncio.run(provider.synthesize("你好世界", "alloy"))

    assert result == fake_audio
    instance.audio.speech.create.assert_called_once()
    call_kwargs = instance.audio.speech.create.call_args[1]
    assert call_kwargs["model"] == "tts-1-hd"
    assert call_kwargs["voice"] == "alloy"
    assert call_kwargs["input"] == "你好世界"
