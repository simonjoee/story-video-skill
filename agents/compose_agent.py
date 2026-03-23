import json
import os
import subprocess
from agents.base import BaseAgent
from schemas.project import VideoProject
from schemas.config import AppConfig


class ComposeAgent(BaseAgent[VideoProject, str]):
    def __init__(self, config: AppConfig, output_dir: str, remotion_dir: str):
        super().__init__(config, output_dir)
        self.remotion_dir = remotion_dir

    async def run(self, input_data: VideoProject) -> str:
        if os.path.exists(input_data.output_path):
            return input_data.output_path

        props = self._build_props(input_data)
        props_path = os.path.join(self.remotion_dir, "input-props.json")
        os.makedirs(self.remotion_dir, exist_ok=True)
        with open(props_path, "w", encoding="utf-8") as f:
            json.dump(props, f, ensure_ascii=False, indent=2)

        output_abs = os.path.abspath(input_data.output_path)

        result = subprocess.run(
            [
                "npx", "remotion", "render",
                "src/index.ts", "StoryVideo",
                f"--props=./input-props.json",
                f"--output={output_abs}",
                "--codec=h264",
            ],
            cwd=self.remotion_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Remotion render failed (exit code {result.returncode}):\n{result.stderr}"
            )

        return input_data.output_path

    def _build_props(self, project: VideoProject) -> dict:
        frames = []
        for frame in project.storyboard.frames:
            image = next(
                (img for img in project.media.images if img.frame_id == frame.frame_id),
                None,
            )
            audio = next(
                (aud for aud in project.media.audios if aud.frame_id == frame.frame_id),
                None,
            )
            frames.append({
                "frameId": frame.frame_id,
                "imagePath": image.file_path if image else "",
                "audioPath": audio.file_path if audio else "",
                "durationSeconds": audio.duration_seconds if audio else frame.duration_seconds,
                "narrationText": frame.narration_text,
                "transition": frame.transition,
            })

        return {
            "fps": project.fps,
            "width": project.resolution[0],
            "height": project.resolution[1],
            "frames": frames,
        }
