import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from orchestrator import run_pipeline
from schemas.config import AppConfig
from schemas.story import PolishedStory
from schemas.storyboard import Storyboard, Frame
from schemas.media import ImageAsset, AudioAsset


def make_config(tmp_path):
    return AppConfig.model_validate({
        "providers": {
            "llm": {"type": "openai", "model": "gpt-4o", "api_key_env": "K"},
            "image": {"type": "openai", "model": "dall-e-3", "api_key_env": "K"},
            "audio": {"type": "openai", "model": "tts-1-hd", "api_key_env": "K", "voice": "alloy"},
        },
        "pipeline": {"max_concurrency": 2, "retry_attempts": 1, "request_timeout": 60},
        "output": {"dir": str(tmp_path / "output"), "resolution": [1920, 1080], "fps": 30},
    })


def test_run_pipeline_full(tmp_path):
    config = make_config(tmp_path)
    output_dir = str(tmp_path / "output")
    remotion_dir = str(tmp_path / "remotion")
    os.makedirs(remotion_dir, exist_ok=True)

    mock_llm = AsyncMock()
    mock_image = AsyncMock()
    mock_audio = AsyncMock()

    mock_llm.complete = AsyncMock(side_effect=[
        json.dumps({
            "title": "T", "full_text": "字" * 250,
            "summary": "S", "tone": "warm",
        }),
        json.dumps({
            "title": "T", "global_style": "cinematic",
            "frames": [{
                "frame_id": 1, "scene_description": "scene",
                "narration_text": "旁白", "duration_seconds": 8.0,
                "visual_style": "cinematic", "transition": "fade",
            }],
        }),
    ])

    mock_image.generate = AsyncMock(return_value=b"\x89PNG")
    mock_audio.synthesize = AsyncMock(return_value=b"\xff\xfb" * 50)

    def fake_render(*args, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, "final.mp4")
        with open(final_path, "wb") as f:
            f.write(b"\x00")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        return mock_result

    with patch("orchestrator.create_llm_provider", return_value=mock_llm), \
         patch("orchestrator.create_image_provider", return_value=mock_image), \
         patch("orchestrator.create_audio_provider", return_value=mock_audio), \
         patch("agents.audio_agent.get_audio_duration", return_value=8.5), \
         patch("agents.compose_agent.subprocess.run", side_effect=fake_render):

        result = asyncio.run(run_pipeline(
            input_text="小明在雨中奔跑",
            config=config,
            remotion_dir=remotion_dir,
        ))

    assert result == os.path.join(output_dir, "final.mp4")
    assert mock_llm.complete.call_count == 2
    assert mock_image.generate.call_count == 1
    assert mock_audio.synthesize.call_count == 1
