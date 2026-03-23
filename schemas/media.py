from pydantic import BaseModel


class ImageAsset(BaseModel):
    frame_id: int
    file_path: str
    width: int
    height: int


class AudioAsset(BaseModel):
    frame_id: int
    file_path: str
    duration_seconds: float


class MediaBundle(BaseModel):
    images: list[ImageAsset]
    audios: list[AudioAsset]
