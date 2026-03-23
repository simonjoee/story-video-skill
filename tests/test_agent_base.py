import asyncio
import json
import os
import pytest
from pydantic import BaseModel
from agents.base import BaseAgent
from schemas.config import AppConfig


class SampleOutput(BaseModel):
    value: str


class SampleAgent(BaseAgent[str, SampleOutput]):
    async def run(self, input_data: str) -> SampleOutput:
        return SampleOutput(value=input_data.upper())


def make_config() -> AppConfig:
    return AppConfig.model_validate({
        "providers": {"llm": {"type": "openai", "model": "m", "api_key_env": "K"}},
    })


def test_agent_save_checkpoint(tmp_path):
    agent = SampleAgent(config=make_config(), output_dir=str(tmp_path))
    data = SampleOutput(value="hello")
    agent.save_checkpoint("test_stage", data)
    path = tmp_path / "test_stage.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["value"] == "hello"


def test_agent_load_checkpoint(tmp_path):
    agent = SampleAgent(config=make_config(), output_dir=str(tmp_path))
    data = SampleOutput(value="world")
    agent.save_checkpoint("test_stage", data)
    restored = agent.load_checkpoint("test_stage", SampleOutput)
    assert restored is not None
    assert restored.value == "world"


def test_agent_load_checkpoint_missing(tmp_path):
    agent = SampleAgent(config=make_config(), output_dir=str(tmp_path))
    result = agent.load_checkpoint("nonexistent", SampleOutput)
    assert result is None


def test_agent_run(tmp_path):
    agent = SampleAgent(config=make_config(), output_dir=str(tmp_path))
    result = asyncio.run(agent.run("hello"))
    assert result.value == "HELLO"
