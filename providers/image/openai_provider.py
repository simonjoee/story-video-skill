import base64
from openai import AsyncOpenAI


class OpenAIImageProvider:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str, style: str, width: int, height: int) -> bytes:
        size = self._resolve_size(width, height)
        response = await self.client.images.generate(
            model=self.model,
            prompt=f"{style}, {prompt}",
            n=1,
            size=size,
            response_format="b64_json",
        )
        b64_data = response.data[0].b64_json
        return base64.b64decode(b64_data)

    def _resolve_size(self, width: int, height: int) -> str:
        if width >= 1792 or height >= 1792:
            return "1792x1024" if width >= height else "1024x1792"
        return "1024x1024"
