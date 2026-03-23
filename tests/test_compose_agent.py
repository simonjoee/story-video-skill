import asyncio
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from agents.compose_agent import ComposeAgent
from schemas.storyboard import Frame, Storyboard
from schemas.media import ImageAsset, AudioAsset, MediaBundle
from schemas.project import VideoProject
from schemas.config import AppConfig


def make_config():
    return AppConfig.model_validate({
        "providers": {},
        "output": {"dir": "./output", "resolution": [1920, 1080], "fps": 30},
    })


def make_project(tmp_path):
    frames = [Frame(frame_id=1, scene_description="s", narration_text="n",
                    duration_seconds=8.0, visual_style="v", transition="fade")]
    sb = Storyboard(title="T", global_style="g", frames=frames, total_duration_seconds=8.0)
    images = [ImageAsset(frame_id=1, file_path=str(tmp_path / "img.png"), width=1920, height=1080)]
    audios = [AudioAsset(frame_id=1, file_path=str(tmp_path / "aud.mp3"), duration_seconds=8.0)]
    return VideoProject(
        storyboard=sb,
        media=MediaBundle(images=images, audios=audios),
        output_path=str(tmp_path / "final.mp4"),
        resolution=[1920, 1080],
        fps=30,
    )


def test_compose_agent_writes_input_props(tmp_path):
    project = make_project(tmp_path)
    remotion_dir = tmp_path / "remotion"
    remotion_dir.mkdir()

    def fake_render(*args, **kwargs):
        (tmp_path / "final.mp4").write_bytes(b"\x00")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Rendered successfully"
        mock_result.stderr = ""
        return mock_result

    with patch("agents.compose_agent.subprocess.run", side_effect=fake_render):
        agent = ComposeAgent(
            config=make_config(), output_dir=str(tmp_path),
            remotion_dir=str(remotion_dir),
        )
        result = asyncio.run(agent.run(project))

    props_path = remotion_dir / "input-props.json"
    assert props_path.exists()
    props = json.loads(props_path.read_text())
    assert props["fps"] == 30
    assert props["width"] == 1920
    assert len(props["frames"]) == 1
    assert props["frames"][0]["frameId"] == 1


def test_compose_agent_calls_remotion_cli(tmp_path):
    project = make_project(tmp_path)
    remotion_dir = tmp_path / "remotion"
    remotion_dir.mkdir()

    def fake_render(*args, **kwargs):
        (tmp_path / "final.mp4").write_bytes(b"\x00")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        return mock_result

    with patch("agents.compose_agent.subprocess.run", side_effect=fake_render):
        agent = ComposeAgent(
            config=make_config(), output_dir=str(tmp_path),
            remotion_dir=str(remotion_dir),
        )
        result = asyncio.run(agent.run(project))

    assert result == str(tmp_path / "final.mp4")


def test_compose_agent_raises_on_render_failure(tmp_path):
    project = make_project(tmp_path)
    remotion_dir = tmp_path / "remotion"
    remotion_dir.mkdir()

    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = ""
    mock_process.stderr = "Error: Composition not found"

    with patch("agents.compose_agent.subprocess.run", return_value=mock_process):
        agent = ComposeAgent(
            config=make_config(), output_dir=str(tmp_path),
            remotion_dir=str(remotion_dir),
        )
        with pytest.raises(RuntimeError, match="Remotion render failed"):
            asyncio.run(agent.run(project))
