from openai import AsyncOpenAI


class OpenAIAudioProvider:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def synthesize(self, text: str, voice: str) -> bytes:
        response = await self.client.audio.speech.create(
            model=self.model,
            voice=voice,
            input=text,
            response_format="mp3",
        )
        return response.content
