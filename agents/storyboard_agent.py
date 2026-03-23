import json
from agents.base import BaseAgent
from schemas.story import PolishedStory
from schemas.storyboard import Storyboard
from schemas.config import AppConfig
from providers.base import LLMProvider

SYSTEM_PROMPT = """你是一位专业的分镜师。请将以下故事分解为视频分镜脚本。
要求：
1. 将故事分为 {n} 个场景帧
2. 每帧的 scene_description 用英文描述画面（供 AI 绘图），包含具体的人物、环境，光线、构图、情绪
3. 每帧的 narration_text 是该段的中文旁白
4. 设定统一的 global_style 确保视觉一致
5. 每帧时长 5-15 秒，总时长约 {duration} 秒
6. transition 可选: fade, cut, dissolve
7. 以 JSON 格式返回: {{"title": "标题", "global_style": "全局风格", "frames": [{{"frame_id": 1, "scene_description": "...", "narration_text": "...", "duration_seconds": 8.0, "visual_style": "...", "transition": "fade"}}]}}
只返回 JSON，不要添加其他内容。"""


class StoryboardAgent(BaseAgent[PolishedStory, Storyboard]):
    def __init__(self, config: AppConfig, output_dir: str, llm: LLMProvider):
        super().__init__(config, output_dir)
        self.llm = llm

    async def run(self, input_data: PolishedStory) -> Storyboard:
        cached = self.load_checkpoint("storyboard", Storyboard)
        if cached is not None:
            return cached

        duration = input_data.target_duration_seconds
        n_frames = max(3, duration // 8)

        system = SYSTEM_PROMPT.format(n=n_frames, duration=duration)
        user_prompt = f"标题: {input_data.title}\n基调: {input_data.tone}\n\n{input_data.full_text}"
        response = await self.llm.complete(system, user_prompt)

        data = json.loads(response)
        frames = data.get("frames", [])
        total = sum(f["duration_seconds"] for f in frames)
        data["total_duration_seconds"] = total

        storyboard = Storyboard.model_validate(data)
        self.save_checkpoint("storyboard", storyboard)
        return storyboard
