import asyncio
import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from providers.image.openai_provider import OpenAIImageProvider


def test_openai_image_provider_generate():
    fake_image_bytes = b"\x89PNG\r\n\x1a\n"
    fake_b64 = base64.b64encode(fake_image_bytes).decode()

    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].b64_json = fake_b64

    with patch("providers.image.openai_provider.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.images.generate = AsyncMock(return_value=mock_response)

        provider = OpenAIImageProvider(model="dall-e-3", api_key="test-key")
        result = asyncio.run(provider.generate(
            prompt="a boy in rain",
            style="cinematic",
            width=1920,
            height=1080,
        ))

    assert result == fake_image_bytes
    instance.images.generate.assert_called_once()
    call_kwargs = instance.images.generate.call_args[1]
    assert call_kwargs["model"] == "dall-e-3"
    assert "cinematic" in call_kwargs["prompt"]
    assert call_kwargs["response_format"] == "b64_json"
