import asyncio
import os
import shutil
import yaml

from schemas.config import AppConfig
from schemas.media import MediaBundle
from schemas.project import VideoProject
from agents.polish_agent import PolishAgent
from agents.storyboard_agent import StoryboardAgent
from agents.image_agent import ImageAgent
from agents.audio_agent import AudioAgent
from agents.compose_agent import ComposeAgent
from providers import create_llm_provider, create_image_provider, create_audio_provider


async def run_pipeline(
    input_text: str,
    config: AppConfig,
    remotion_dir: str = "remotion",
    resume: bool = False,
) -> str:
    output_dir = config.output.dir

    if not resume and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    llm = create_llm_provider(config.providers["llm"])
    image_provider = create_image_provider(config.providers["image"])
    audio_provider = create_audio_provider(config.providers["audio"])

    print("[1/5] Polish Agent...", flush=True)
    story = await PolishAgent(config, output_dir, llm=llm).run(input_text)
    print(f"[1/5] Polish Agent... done (title: {story.title})", flush=True)

    print("[2/5] Storyboard Agent...", flush=True)
    storyboard = await StoryboardAgent(config, output_dir, llm=llm).run(story)
    print(f"[2/5] Storyboard Agent... done ({len(storyboard.frames)} frames, {storyboard.total_duration_seconds}s)", flush=True)

    print("[3/5] Image Agent + Audio Agent (parallel)...", flush=True)
    image_agent = ImageAgent(config, output_dir, image_provider=image_provider)
    audio_agent = AudioAgent(config, output_dir, audio_provider=audio_provider)
    images, audios = await asyncio.gather(
        image_agent.run(storyboard),
        audio_agent.run(storyboard),
    )
    print(f"[3/5] Image + Audio done ({len(images)} images, {len(audios)} audios)", flush=True)

    print("[4/5] Compose Agent (Remotion rendering)...", flush=True)
    project = VideoProject(
        storyboard=storyboard,
        media=MediaBundle(images=images, audios=audios),
        output_path=os.path.join(os.path.abspath(output_dir), "final.mp4"),
        resolution=config.output.resolution,
        fps=config.output.fps,
    )
    video_path = await ComposeAgent(config, output_dir, remotion_dir=remotion_dir).run(project)
    print(f"[5/5] Done! -> {video_path}", flush=True)

    return video_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Story Video Pipeline")
    parser.add_argument("text", help="Input text or path to text file")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--remotion-dir", default="remotion", help="Path to Remotion project")
    args = parser.parse_args()

    input_text = open(args.text).read() if os.path.isfile(args.text) else args.text

    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    result = asyncio.run(run_pipeline(
        input_text=input_text,
        config=config,
        remotion_dir=args.remotion_dir,
        resume=args.resume,
    ))
    print(f"\nVideo generated: {result}")
