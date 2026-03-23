import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from providers.llm.openai_provider import OpenAILLMProvider


def test_openai_llm_provider_complete():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"title": "Test Story"}'

    with patch("providers.llm.openai_provider.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAILLMProvider(
            model="gpt-4o",
            api_key="test-key",
        )
        result = asyncio.run(provider.complete("system prompt", "user prompt"))

    assert result == '{"title": "Test Story"}'
    instance.chat.completions.create.assert_called_once()
    call_kwargs = instance.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"
