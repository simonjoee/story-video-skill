from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...


@runtime_checkable
class ImageProvider(Protocol):
    async def generate(self, prompt: str, style: str, width: int, height: int) -> bytes: ...


@runtime_checkable
class AudioProvider(Protocol):
    async def synthesize(self, text: str, voice: str) -> bytes: ...
