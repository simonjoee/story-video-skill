from anthropic import AsyncAnthropic


class AnthropicLLMProvider:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.content[0].text
