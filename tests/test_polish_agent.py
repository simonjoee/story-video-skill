import asyncio
import json
import pytest
from unittest.mock import AsyncMock
from agents.polish_agent import PolishAgent
from schemas.story import PolishedStory
from schemas.config import AppConfig


def make_config():
    return AppConfig.model_validate({
        "providers": {"llm": {"type": "openai", "model": "gpt-4o", "api_key_env": "K"}},
    })


def test_polish_agent_run(tmp_path):
    expected_response = json.dumps({
        "title": "雨中的少年",
        "full_text": "在一个宁静的小镇上，" + "故事内容" * 50,
        "summary": "少年在雨中奔跑的故事",
        "tone": "warm",
    })

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=expected_response)

    agent = PolishAgent(config=make_config(), output_dir=str(tmp_path), llm=mock_llm)
    result = asyncio.run(agent.run("小明在雨中奔跑"))

    assert isinstance(result, PolishedStory)
    assert result.title == "雨中的少年"
    assert result.tone == "warm"
    assert result.target_duration_seconds > 0
    mock_llm.complete.assert_called_once()


def test_polish_agent_checkpoint_resume(tmp_path):
    existing = PolishedStory(
        title="Cached", full_text="text", summary="s", tone="warm",
        target_duration_seconds=60,
    )
    (tmp_path / "polish.json").write_text(existing.model_dump_json())

    mock_llm = AsyncMock()
    agent = PolishAgent(config=make_config(), output_dir=str(tmp_path), llm=mock_llm)
    result = asyncio.run(agent.run("anything"))

    assert result.title == "Cached"
    mock_llm.complete.assert_not_called()


def test_polish_agent_duration_estimation(tmp_path):
    text_500_chars = "字" * 500
    expected_response = json.dumps({
        "title": "T", "full_text": text_500_chars, "summary": "S", "tone": "warm",
    })

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=expected_response)

    agent = PolishAgent(config=make_config(), output_dir=str(tmp_path), llm=mock_llm)
    result = asyncio.run(agent.run("input"))

    assert result.target_duration_seconds == 120
