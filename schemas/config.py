from __future__ import annotations
from pydantic import BaseModel, model_validator


class ProviderConfig(BaseModel):
    type: str
    model: str
    api_key_env: str
    extra: dict = {}

    @model_validator(mode="before")
    @classmethod
    def collect_extra_fields(cls, values: dict) -> dict:
        known_fields = {"type", "model", "api_key_env", "extra"}
        extra = {k: v for k, v in values.items() if k not in known_fields}
        filtered = {k: v for k, v in values.items() if k in known_fields}
        filtered["extra"] = {**filtered.get("extra", {}), **extra}
        return filtered


class PipelineConfig(BaseModel):
    max_concurrency: int = 3
    retry_attempts: int = 3
    request_timeout: int = 60


class OutputConfig(BaseModel):
    dir: str = "./output"
    resolution: list[int] = [1920, 1080]
    fps: int = 30


class AppConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    pipeline: PipelineConfig = PipelineConfig()
    output: OutputConfig = OutputConfig()
