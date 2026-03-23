import asyncio
import pytest
from providers.base import LLMProvider, ImageProvider, AudioProvider


class MockLLM:
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return '{"title": "test"}'


class MockImage:
    async def generate(self, prompt: str, style: str, width: int, height: int) -> bytes:
        return b"\x89PNG"


class MockAudio:
    async def synthesize(self, text: str, voice: str) -> bytes:
        return b"\xff\xfb"


def test_mock_llm_satisfies_protocol():
    provider: LLMProvider = MockLLM()
    result = asyncio.run(provider.complete("sys", "user"))
    assert isinstance(result, str)


def test_mock_image_satisfies_protocol():
    provider: ImageProvider = MockImage()
    result = asyncio.run(provider.generate("prompt", "style", 1920, 1080))
    assert isinstance(result, bytes)


def test_mock_audio_satisfies_protocol():
    provider: AudioProvider = MockAudio()
    result = asyncio.run(provider.synthesize("hello", "alloy"))
    assert isinstance(result, bytes)
