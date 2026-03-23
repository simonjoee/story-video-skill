import asyncio
import os
from pydub import AudioSegment
from agents.base import BaseAgent
from schemas.storyboard import Storyboard
from schemas.media import AudioAsset
from schemas.config import AppConfig
from providers.base import AudioProvider


def get_audio_duration(file_path: str) -> float:
    audio = AudioSegment.from_file(file_path)
    return audio.duration_seconds


class AudioAgent(BaseAgent[Storyboard, list[AudioAsset]]):
    def __init__(self, config: AppConfig, output_dir: str, audio_provider: AudioProvider):
        super().__init__(config, output_dir)
        self.audio_provider = audio_provider
        self.audios_dir = os.path.join(output_dir, "audios")
        os.makedirs(self.audios_dir, exist_ok=True)

    async def run(self, input_data: Storyboard) -> list[AudioAsset]:
        semaphore = asyncio.Semaphore(self.config.pipeline.max_concurrency)
        voice_cfg = self.config.providers.get("audio", None)
        voice_name = voice_cfg.extra.get("voice", "alloy") if voice_cfg else "alloy"

        async def generate_frame(frame) -> AudioAsset:
            file_path = os.path.join(self.audios_dir, f"frame_{frame.frame_id}.mp3")

            if os.path.exists(file_path):
                duration = get_audio_duration(file_path)
                return AudioAsset(
                    frame_id=frame.frame_id,
                    file_path=os.path.abspath(file_path),
                    duration_seconds=duration,
                )

            async with semaphore:
                audio_bytes = await self._synthesize_with_retry(
                    frame.narration_text, voice_name,
                )

            with open(file_path, "wb") as f:
                f.write(audio_bytes)

            duration = get_audio_duration(file_path)
            return AudioAsset(
                frame_id=frame.frame_id,
                file_path=os.path.abspath(file_path),
                duration_seconds=duration,
            )

        tasks = [generate_frame(frame) for frame in input_data.frames]
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda r: r.frame_id)

    async def _synthesize_with_retry(self, text: str, voice: str) -> bytes:
        max_retries = self.config.pipeline.retry_attempts
        for attempt in range(max_retries):
            try:
                return await self.audio_provider.synthesize(text=text, voice=voice)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
