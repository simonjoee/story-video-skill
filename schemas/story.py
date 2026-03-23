from pydantic import BaseModel


class PolishedStory(BaseModel):
    title: str
    full_text: str
    summary: str
    tone: str
    target_duration_seconds: int
