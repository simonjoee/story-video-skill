import json
import pytest
from schemas.story import PolishedStory
from schemas.storyboard import Frame, Storyboard
from schemas.media import ImageAsset, AudioAsset, MediaBundle
from schemas.project import VideoProject


def test_polished_story_creation():
    story = PolishedStory(
        title="The Rain Runner",
        full_text="A" * 500,
        summary="A boy runs in the rain.",
        tone="warm",
        target_duration_seconds=120,
    )
    assert story.title == "The Rain Runner"
    assert story.target_duration_seconds == 120


def test_polished_story_json_roundtrip():
    story = PolishedStory(
        title="Test", full_text="text", summary="sum", tone="warm",
        target_duration_seconds=60,
    )
    data = story.model_dump_json()
    restored = PolishedStory.model_validate_json(data)
    assert restored == story


def test_frame_creation():
    frame = Frame(
        frame_id=1,
        scene_description="A boy running in rain",
        narration_text="小明在雨中奔跑",
        duration_seconds=8.0,
        visual_style="cinematic",
        transition="fade",
    )
    assert frame.frame_id == 1
    assert frame.transition == "fade"


def test_storyboard_creation():
    frames = [
        Frame(frame_id=i, scene_description=f"scene {i}",
              narration_text=f"旁白 {i}", duration_seconds=8.0,
              visual_style="cinematic", transition="fade")
        for i in range(1, 4)
    ]
    sb = Storyboard(
        title="Test", global_style="cinematic realism",
        frames=frames, total_duration_seconds=24.0,
    )
    assert len(sb.frames) == 3
    assert sb.total_duration_seconds == 24.0


def test_media_bundle():
    images = [ImageAsset(frame_id=1, file_path="/out/frame_1.png", width=1920, height=1080)]
    audios = [AudioAsset(frame_id=1, file_path="/out/frame_1.mp3", duration_seconds=8.2)]
    bundle = MediaBundle(images=images, audios=audios)
    assert len(bundle.images) == 1
    assert bundle.audios[0].duration_seconds == 8.2


def test_video_project():
    frames = [Frame(frame_id=1, scene_description="s", narration_text="n",
                    duration_seconds=8.0, visual_style="v", transition="fade")]
    sb = Storyboard(title="T", global_style="g", frames=frames, total_duration_seconds=8.0)
    images = [ImageAsset(frame_id=1, file_path="/img.png", width=1920, height=1080)]
    audios = [AudioAsset(frame_id=1, file_path="/aud.mp3", duration_seconds=8.0)]
    project = VideoProject(
        storyboard=sb,
        media=MediaBundle(images=images, audios=audios),
        output_path="/out/final.mp4",
        resolution=[1920, 1080],
        fps=30,
    )
    assert project.fps == 30
    assert project.resolution == [1920, 1080]


def test_video_project_json_roundtrip():
    frames = [Frame(frame_id=1, scene_description="s", narration_text="n",
                    duration_seconds=8.0, visual_style="v", transition="fade")]
    sb = Storyboard(title="T", global_style="g", frames=frames, total_duration_seconds=8.0)
    images = [ImageAsset(frame_id=1, file_path="/img.png", width=1920, height=1080)]
    audios = [AudioAsset(frame_id=1, file_path="/aud.mp3", duration_seconds=8.0)]
    project = VideoProject(
        storyboard=sb,
        media=MediaBundle(images=images, audios=audios),
        output_path="/out/final.mp4",
        resolution=[1920, 1080],
        fps=30,
    )
    data = project.model_dump_json()
    restored = VideoProject.model_validate_json(data)
    assert restored == project
