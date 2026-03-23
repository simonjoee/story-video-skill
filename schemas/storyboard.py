from pydantic import BaseModel


class Frame(BaseModel):
    frame_id: int
    scene_description: str
    narration_text: str
    duration_seconds: float
    visual_style: str
    transition: str


class Storyboard(BaseModel):
    title: str
    global_style: str
    frames: list[Frame]
    total_duration_seconds: float
