import asyncio
import json
import pytest
from unittest.mock import AsyncMock
from agents.storyboard_agent import StoryboardAgent
from schemas.story import PolishedStory
from schemas.storyboard import Storyboard
from schemas.config import AppConfig


def make_config():
    return AppConfig.model_validate({
        "providers": {"llm": {"type": "openai", "model": "gpt-4o", "api_key_env": "K"}},
    })


def make_story():
    return PolishedStory(
        title="雨中的少年",
        full_text="在一个宁静的小镇上..." + "故事" * 100,
        summary="少年在雨中奔跑",
        tone="warm",
        target_duration_seconds=96,
    )


def test_storyboard_agent_run(tmp_path):
    llm_response = json.dumps({
        "title": "雨中的少年",
        "global_style": "cinematic realism, soft lighting",
        "frames": [
            {
                "frame_id": 1,
                "scene_description": "A quiet small town with cobblestone streets under grey sky",
                "narration_text": "在一个宁静的小镇上",
                "duration_seconds": 8.0,
                "visual_style": "cinematic",
                "transition": "fade",
            },
            {
                "frame_id": 2,
                "scene_description": "A boy starts running in the rain",
                "narration_text": "少年开始在雨中奔跑",
                "duration_seconds": 10.0,
                "visual_style": "cinematic",
                "transition": "dissolve",
            },
        ],
    })

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=llm_response)

    agent = StoryboardAgent(config=make_config(), output_dir=str(tmp_path), llm=mock_llm)
    result = asyncio.run(agent.run(make_story()))

    assert isinstance(result, Storyboard)
    assert len(result.frames) == 2
    assert result.global_style == "cinematic realism, soft lighting"
    assert result.total_duration_seconds == 18.0
    assert result.frames[0].scene_description.startswith("A quiet")


def test_storyboard_agent_checkpoint_resume(tmp_path):
    frames_data = [{
        "frame_id": 1, "scene_description": "s", "narration_text": "n",
        "duration_seconds": 8.0, "visual_style": "v", "transition": "fade",
    }]
    existing = Storyboard(
        title="Cached", global_style="g", frames=frames_data,
        total_duration_seconds=8.0,
    )
    (tmp_path / "storyboard.json").write_text(existing.model_dump_json())

    mock_llm = AsyncMock()
    agent = StoryboardAgent(config=make_config(), output_dir=str(tmp_path), llm=mock_llm)
    result = asyncio.run(agent.run(make_story()))

    assert result.title == "Cached"
    mock_llm.complete.assert_not_called()
