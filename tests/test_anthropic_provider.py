import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from providers.llm.anthropic_provider import AnthropicLLMProvider


def test_anthropic_llm_provider_complete():
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = '{"title": "Test Story"}'

    with patch("providers.llm.anthropic_provider.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicLLMProvider(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        )
        result = asyncio.run(provider.complete("system prompt", "user prompt"))

    assert result == '{"title": "Test Story"}'
    instance.messages.create.assert_called_once()
    call_kwargs = instance.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"
    assert call_kwargs["system"] == "system prompt"
