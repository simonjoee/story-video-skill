from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pydantic import BaseModel
from schemas.config import AppConfig

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class BaseAgent(ABC, Generic[TIn, TOut]):
    def __init__(self, config: AppConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    async def run(self, input_data: TIn) -> TOut: ...

    def save_checkpoint(self, stage: str, data: BaseModel) -> None:
        path = os.path.join(self.output_dir, f"{stage}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(data.model_dump_json(indent=2))

    def load_checkpoint(self, stage: str, model_cls: type[BaseModel]) -> BaseModel | None:
        path = os.path.join(self.output_dir, f"{stage}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return model_cls.model_validate_json(f.read())
        return None
