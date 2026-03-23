from openai import AsyncOpenAI


class OpenAILLMProvider:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
