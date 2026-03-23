import json
from agents.base import BaseAgent
from schemas.story import PolishedStory
from schemas.config import AppConfig
from providers.base import LLMProvider

SYSTEM_PROMPT = """你是一位专业的故事编剧。请将以下短文案润色扩写为一个完整的故事。
要求：
1. 保持原文的核心主题和情感
2. 添加生动的细节、对话和场景描写
3. 目标字数约 {target_words} 字
4. 故事结构清晰：开头/发展/高潮/结尾
5. 以 JSON 格式返回: {{"title": "标题", "full_text": "完整故事", "summary": "一句话摘要", "tone": "基调"}}
只返回 JSON，不要添加其他内容。"""


class PolishAgent(BaseAgent[str, PolishedStory]):
    def __init__(self, config: AppConfig, output_dir: str, llm: LLMProvider):
        super().__init__(config, output_dir)
        self.llm = llm

    async def run(self, input_data: str) -> PolishedStory:
        cached = self.load_checkpoint("polish", PolishedStory)
        if cached is not None:
            return cached

        target_words = max(300, len(input_data) * 5)

        system = SYSTEM_PROMPT.format(target_words=target_words)
        response = await self.llm.complete(system, input_data)

        data = json.loads(response)
        full_text = data["full_text"]
        duration = int(len(full_text) / 250 * 60)
        data["target_duration_seconds"] = duration

        story = PolishedStory.model_validate(data)
        self.save_checkpoint("polish", story)
        return story
