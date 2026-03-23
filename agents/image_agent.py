import asyncio
import os
from agents.base import BaseAgent
from schemas.storyboard import Storyboard
from schemas.media import ImageAsset
from schemas.config import AppConfig
from providers.base import ImageProvider


class ImageAgent(BaseAgent[Storyboard, list[ImageAsset]]):
    def __init__(self, config: AppConfig, output_dir: str, image_provider: ImageProvider):
        super().__init__(config, output_dir)
        self.image_provider = image_provider
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    async def run(self, input_data: Storyboard) -> list[ImageAsset]:
        semaphore = asyncio.Semaphore(self.config.pipeline.max_concurrency)
        width, height = self.config.output.resolution

        async def generate_frame(frame) -> ImageAsset:
            file_path = os.path.join(self.images_dir, f"frame_{frame.frame_id}.png")

            if os.path.exists(file_path):
                return ImageAsset(
                    frame_id=frame.frame_id,
                    file_path=os.path.abspath(file_path),
                    width=width, height=height,
                )

            prompt = f"{input_data.global_style} style, {frame.scene_description}, {frame.visual_style}"

            async with semaphore:
                image_bytes = await self._generate_with_retry(prompt, frame.visual_style, width, height)

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            return ImageAsset(
                frame_id=frame.frame_id,
                file_path=os.path.abspath(file_path),
                width=width, height=height,
            )

        tasks = [generate_frame(frame) for frame in input_data.frames]
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda r: r.frame_id)

    async def _generate_with_retry(self, prompt: str, style: str, width: int, height: int) -> bytes:
        max_retries = self.config.pipeline.retry_attempts
        for attempt in range(max_retries):
            try:
                return await self.image_provider.generate(
                    prompt=prompt, style=style, width=width, height=height,
                )
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
