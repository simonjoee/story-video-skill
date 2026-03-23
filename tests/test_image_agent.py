import asyncio
import os
import pytest
from unittest.mock import AsyncMock
from agents.image_agent import ImageAgent
from schemas.storyboard import Frame, Storyboard
from schemas.media import ImageAsset
from schemas.config import AppConfig


def make_config():
    return AppConfig.model_validate({
        "providers": {"image": {"type": "openai", "model": "dall-e-3", "api_key_env": "K"}},
        "pipeline": {"max_concurrency": 2, "retry_attempts": 1, "request_timeout": 60},
        "output": {"resolution": [1920, 1080]},
    })


def make_storyboard():
    frames = [
        Frame(frame_id=1, scene_description="A boy in rain",
              narration_text="旁白1", duration_seconds=8.0,
              visual_style="cinematic", transition="fade"),
        Frame(frame_id=2, scene_description="A rainbow appears",
              narration_text="旁白2", duration_seconds=10.0,
              visual_style="cinematic", transition="dissolve"),
    ]
    return Storyboard(title="T", global_style="cinematic realism",
                      frames=frames, total_duration_seconds=18.0)


def test_image_agent_generates_all_frames(tmp_path):
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    mock_image = AsyncMock()
    mock_image.generate = AsyncMock(return_value=fake_png)

    agent = ImageAgent(config=make_config(), output_dir=str(tmp_path), image_provider=mock_image)
    result = asyncio.run(agent.run(make_storyboard()))

    assert len(result) == 2
    assert all(isinstance(r, ImageAsset) for r in result)
    assert os.path.exists(result[0].file_path)
    assert os.path.exists(result[1].file_path)
    assert mock_image.generate.call_count == 2


def test_image_agent_skips_existing_files(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame_1.png").write_bytes(b"\x89PNG")

    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    mock_image = AsyncMock()
    mock_image.generate = AsyncMock(return_value=fake_png)

    agent = ImageAgent(config=make_config(), output_dir=str(tmp_path), image_provider=mock_image)
    result = asyncio.run(agent.run(make_storyboard()))

    assert len(result) == 2
    assert mock_image.generate.call_count == 1


def test_image_agent_prompt_includes_global_style(tmp_path):
    fake_png = b"\x89PNG"
    mock_image = AsyncMock()
    mock_image.generate = AsyncMock(return_value=fake_png)

    agent = ImageAgent(config=make_config(), output_dir=str(tmp_path), image_provider=mock_image)
    asyncio.run(agent.run(make_storyboard()))

    call_args = mock_image.generate.call_args_list[0]
    prompt = call_args[1]["prompt"] if "prompt" in call_args[1] else call_args[0][0]
    assert "cinematic realism" in prompt
