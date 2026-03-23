from pydantic import BaseModel
from schemas.storyboard import Storyboard
from schemas.media import MediaBundle


class VideoProject(BaseModel):
    storyboard: Storyboard
    media: MediaBundle
    output_path: str
    resolution: list[int] = [1920, 1080]
    fps: int = 30
