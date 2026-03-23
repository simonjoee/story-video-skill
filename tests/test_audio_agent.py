import asyncio
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.audio_agent import AudioAgent
from schemas.storyboard import Frame, Storyboard
from schemas.media import AudioAsset
from schemas.config import AppConfig


def make_config():
    return AppConfig.model_validate({
        "providers": {"audio": {"type": "openai", "model": "tts-1-hd", "api_key_env": "K", "voice": "alloy"}},
        "pipeline": {"max_concurrency": 2, "retry_attempts": 1, "request_timeout": 60},
    })


def make_storyboard():
    frames = [
        Frame(frame_id=1, scene_description="s1", narration_text="你好世界",
              duration_seconds=8.0, visual_style="v", transition="fade"),
        Frame(frame_id=2, scene_description="s2", narration_text="再见",
              duration_seconds=6.0, visual_style="v", transition="cut"),
    ]
    return Storyboard(title="T", global_style="g", frames=frames, total_duration_seconds=14.0)


def test_audio_agent_generates_all_frames(tmp_path):
    fake_mp3 = b"\xff\xfb\x90\x00" * 100

    mock_audio = AsyncMock()
    mock_audio.synthesize = AsyncMock(return_value=fake_mp3)

    with patch("agents.audio_agent.get_audio_duration", return_value=8.5):
        agent = AudioAgent(config=make_config(), output_dir=str(tmp_path), audio_provider=mock_audio)
        result = asyncio.run(agent.run(make_storyboard()))

    assert len(result) == 2
    assert all(isinstance(r, AudioAsset) for r in result)
    assert result[0].duration_seconds == 8.5
    assert os.path.exists(result[0].file_path)


def test_audio_agent_skips_existing(tmp_path):
    audios_dir = tmp_path / "audios"
    audios_dir.mkdir()
    (audios_dir / "frame_1.mp3").write_bytes(b"\xff\xfb")

    fake_mp3 = b"\xff\xfb\x90\x00" * 100
    mock_audio = AsyncMock()
    mock_audio.synthesize = AsyncMock(return_value=fake_mp3)

    with patch("agents.audio_agent.get_audio_duration", return_value=6.0):
        agent = AudioAgent(config=make_config(), output_dir=str(tmp_path), audio_provider=mock_audio)
        result = asyncio.run(agent.run(make_storyboard()))

    assert len(result) == 2
    assert mock_audio.synthesize.call_count == 1


def test_audio_agent_uses_voice_from_config(tmp_path):
    fake_mp3 = b"\xff\xfb"
    mock_audio = AsyncMock()
    mock_audio.synthesize = AsyncMock(return_value=fake_mp3)

    with patch("agents.audio_agent.get_audio_duration", return_value=5.0):
        agent = AudioAgent(config=make_config(), output_dir=str(tmp_path), audio_provider=mock_audio)
        asyncio.run(agent.run(make_storyboard()))

    call_kwargs = mock_audio.synthesize.call_args_list[0][1]
    assert call_kwargs["voice"] == "alloy"
