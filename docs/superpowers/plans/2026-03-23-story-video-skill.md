# Story Video Skill Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenCode Skill that converts text into narrated video using a 5-agent pipeline (polish, storyboard, image, audio, compose).

**Architecture:** Python orchestrator drives 5 agents in sequence (with parallel image+audio). Each agent uses pluggable providers (LLM, image, audio) via Protocol interfaces. Video composition uses Remotion (TypeScript) invoked as subprocess. All inter-agent data flows through Pydantic schemas.

**Tech Stack:** Python 3.11+, Pydantic v2, asyncio, pydub, PyYAML, openai SDK, anthropic SDK | TypeScript, Remotion v4, React 18

---

## Chunk 1: Project Scaffolding, Schemas, and Config

### Task 1: Project initialization and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `.gitignore`
- Create: `schemas/__init__.py`
- Create: `agents/__init__.py`
- Create: `providers/__init__.py`
- Create: `providers/llm/__init__.py`
- Create: `providers/image/__init__.py`
- Create: `providers/audio/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize Python project with requirements.txt**

```
# requirements.txt
pydantic>=2.0,<3.0
pyyaml>=6.0,<7.0
openai>=1.0,<2.0
anthropic>=0.30,<1.0
pydub>=0.25,<1.0
pytest>=8.0,<9.0
```

- [ ] **Step 2: Create config.yaml with default configuration**

```yaml
providers:
  llm:
    type: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
  image:
    type: openai
    model: dall-e-3
    api_key_env: OPENAI_API_KEY
    default_size: [1920, 1080]
  audio:
    type: openai
    model: tts-1-hd
    voice: alloy
    api_key_env: OPENAI_API_KEY

pipeline:
  max_concurrency: 3
  retry_attempts: 3
  request_timeout: 60

output:
  dir: ./output
  resolution: [1920, 1080]
  fps: 30
```

- [ ] **Step 3: Create .gitignore**

```
output/
__pycache__/
*.pyc
.env
remotion/node_modules/
remotion/dist/
remotion/input-props.json
```

- [ ] **Step 4: Create all __init__.py files**

Create empty `__init__.py` in: `schemas/`, `agents/`, `providers/`, `providers/llm/`, `providers/image/`, `providers/audio/`, `tests/`.

- [ ] **Step 5: Install dependencies and verify**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt config.yaml .gitignore schemas/ agents/ providers/
git commit -m "feat: initialize project scaffolding and dependencies"
```

---

### Task 2: Configuration schema (schemas/config.py)

**Files:**
- Create: `schemas/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
import yaml
from schemas.config import AppConfig, ProviderConfig, PipelineConfig, OutputConfig


def test_app_config_from_valid_yaml():
    raw = {
        "providers": {
            "llm": {"type": "openai", "model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
            "image": {"type": "openai", "model": "dall-e-3", "api_key_env": "OPENAI_API_KEY", "default_size": [1920, 1080]},
            "audio": {"type": "openai", "model": "tts-1-hd", "api_key_env": "OPENAI_API_KEY", "voice": "alloy"},
        },
        "pipeline": {"max_concurrency": 3, "retry_attempts": 3, "request_timeout": 60},
        "output": {"dir": "./output", "resolution": [1920, 1080], "fps": 30},
    }
    config = AppConfig.model_validate(raw)
    assert config.providers["llm"].type == "openai"
    assert config.providers["llm"].model == "gpt-4o"
    assert config.pipeline.max_concurrency == 3
    assert config.output.fps == 30


def test_provider_config_extra_fields():
    """Provider-specific fields like voice, default_size go into extra."""
    raw = {"type": "openai", "model": "tts-1-hd", "api_key_env": "KEY", "voice": "alloy"}
    config = ProviderConfig.model_validate(raw)
    assert config.type == "openai"
    assert config.extra["voice"] == "alloy"


def test_app_config_defaults():
    raw = {
        "providers": {
            "llm": {"type": "openai", "model": "gpt-4o", "api_key_env": "KEY"},
        },
    }
    config = AppConfig.model_validate(raw)
    assert config.output.dir == "./output"
    assert config.output.resolution == [1920, 1080]
    assert config.output.fps == 30
    assert config.pipeline.max_concurrency == 3


def test_load_from_config_yaml_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "providers": {
            "llm": {"type": "openai", "model": "gpt-4o", "api_key_env": "KEY"},
        },
    }))
    with open(config_file) as f:
        raw = yaml.safe_load(f)
    config = AppConfig.model_validate(raw)
    assert config.providers["llm"].type == "openai"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'schemas.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# schemas/config.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add schemas/config.py tests/test_config.py
git commit -m "feat: add AppConfig schema with provider extra field collection"
```

---

### Task 3: Data schemas (story, storyboard, media, project)

**Files:**
- Create: `schemas/story.py`
- Create: `schemas/storyboard.py`
- Create: `schemas/media.py`
- Create: `schemas/project.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schemas.py
import json
import pytest
from schemas.story import PolishedStory
from schemas.storyboard import Frame, Storyboard
from schemas.media import ImageAsset, AudioAsset, MediaBundle
from schemas.project import VideoProject


def test_polished_story_creation():
    story = PolishedStory(
        title="The Rain Runner",
        full_text="A" * 500,
        summary="A boy runs in the rain.",
        tone="warm",
        target_duration_seconds=120,
    )
    assert story.title == "The Rain Runner"
    assert story.target_duration_seconds == 120


def test_polished_story_json_roundtrip():
    story = PolishedStory(
        title="Test", full_text="text", summary="sum", tone="warm",
        target_duration_seconds=60,
    )
    data = story.model_dump_json()
    restored = PolishedStory.model_validate_json(data)
    assert restored == story


def test_frame_creation():
    frame = Frame(
        frame_id=1,
        scene_description="A boy running in rain",
        narration_text="小明在雨中奔跑",
        duration_seconds=8.0,
        visual_style="cinematic",
        transition="fade",
    )
    assert frame.frame_id == 1
    assert frame.transition == "fade"


def test_storyboard_creation():
    frames = [
        Frame(frame_id=i, scene_description=f"scene {i}",
              narration_text=f"旁白 {i}", duration_seconds=8.0,
              visual_style="cinematic", transition="fade")
        for i in range(1, 4)
    ]
    sb = Storyboard(
        title="Test", global_style="cinematic realism",
        frames=frames, total_duration_seconds=24.0,
    )
    assert len(sb.frames) == 3
    assert sb.total_duration_seconds == 24.0


def test_media_bundle():
    images = [ImageAsset(frame_id=1, file_path="/out/frame_1.png", width=1920, height=1080)]
    audios = [AudioAsset(frame_id=1, file_path="/out/frame_1.mp3", duration_seconds=8.2)]
    bundle = MediaBundle(images=images, audios=audios)
    assert len(bundle.images) == 1
    assert bundle.audios[0].duration_seconds == 8.2


def test_video_project():
    frames = [Frame(frame_id=1, scene_description="s", narration_text="n",
                    duration_seconds=8.0, visual_style="v", transition="fade")]
    sb = Storyboard(title="T", global_style="g", frames=frames, total_duration_seconds=8.0)
    images = [ImageAsset(frame_id=1, file_path="/img.png", width=1920, height=1080)]
    audios = [AudioAsset(frame_id=1, file_path="/aud.mp3", duration_seconds=8.0)]
    project = VideoProject(
        storyboard=sb,
        media=MediaBundle(images=images, audios=audios),
        output_path="/out/final.mp4",
        resolution=[1920, 1080],
        fps=30,
    )
    assert project.fps == 30
    assert project.resolution == [1920, 1080]


def test_video_project_json_roundtrip():
    frames = [Frame(frame_id=1, scene_description="s", narration_text="n",
                    duration_seconds=8.0, visual_style="v", transition="fade")]
    sb = Storyboard(title="T", global_style="g", frames=frames, total_duration_seconds=8.0)
    images = [ImageAsset(frame_id=1, file_path="/img.png", width=1920, height=1080)]
    audios = [AudioAsset(frame_id=1, file_path="/aud.mp3", duration_seconds=8.0)]
    project = VideoProject(
        storyboard=sb,
        media=MediaBundle(images=images, audios=audios),
        output_path="/out/final.mp4",
        resolution=[1920, 1080],
        fps=30,
    )
    data = project.model_dump_json()
    restored = VideoProject.model_validate_json(data)
    assert restored == project
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'schemas.story'`

- [ ] **Step 3: Implement all schema files**

```python
# schemas/story.py
from pydantic import BaseModel


class PolishedStory(BaseModel):
    title: str
    full_text: str
    summary: str
    tone: str
    target_duration_seconds: int
```

```python
# schemas/storyboard.py
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
```

```python
# schemas/media.py
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
```

```python
# schemas/project.py
from pydantic import BaseModel
from schemas.storyboard import Storyboard
from schemas.media import MediaBundle


class VideoProject(BaseModel):
    storyboard: Storyboard
    media: MediaBundle
    output_path: str
    resolution: list[int] = [1920, 1080]
    fps: int = 30
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add schemas/story.py schemas/storyboard.py schemas/media.py schemas/project.py tests/test_schemas.py
git commit -m "feat: add all Pydantic data schemas for pipeline stages"
```

---

## Chunk 2: Provider Layer

### Task 4: Provider Protocol definitions (providers/base.py)

**Files:**
- Create: `providers/base.py`
- Create: `tests/test_providers_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_base.py
import asyncio
import pytest
from providers.base import LLMProvider, ImageProvider, AudioProvider


class MockLLM:
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return '{"title": "test"}'


class MockImage:
    async def generate(self, prompt: str, style: str, width: int, height: int) -> bytes:
        return b"\x89PNG"


class MockAudio:
    async def synthesize(self, text: str, voice: str) -> bytes:
        return b"\xff\xfb"


def test_mock_llm_satisfies_protocol():
    provider: LLMProvider = MockLLM()
    result = asyncio.run(provider.complete("sys", "user"))
    assert isinstance(result, str)


def test_mock_image_satisfies_protocol():
    provider: ImageProvider = MockImage()
    result = asyncio.run(provider.generate("prompt", "style", 1920, 1080))
    assert isinstance(result, bytes)


def test_mock_audio_satisfies_protocol():
    provider: AudioProvider = MockAudio()
    result = asyncio.run(provider.synthesize("hello", "alloy"))
    assert isinstance(result, bytes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_providers_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.base'`

- [ ] **Step 3: Write minimal implementation**

```python
# providers/base.py
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...


@runtime_checkable
class ImageProvider(Protocol):
    async def generate(self, prompt: str, style: str, width: int, height: int) -> bytes: ...


@runtime_checkable
class AudioProvider(Protocol):
    async def synthesize(self, text: str, voice: str) -> bytes: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_providers_base.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/base.py tests/test_providers_base.py
git commit -m "feat: add LLM, Image, Audio provider Protocol definitions"
```

---

### Task 5: OpenAI LLM provider (providers/llm/openai_provider.py)

**Files:**
- Create: `providers/llm/openai_provider.py`
- Create: `tests/test_llm_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_provider.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from providers.llm.openai_provider import OpenAILLMProvider


def test_openai_llm_provider_complete():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"title": "Test Story"}'

    with patch("providers.llm.openai_provider.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAILLMProvider(
            model="gpt-4o",
            api_key="test-key",
        )
        result = asyncio.run(provider.complete("system prompt", "user prompt"))

    assert result == '{"title": "Test Story"}'
    instance.chat.completions.create.assert_called_once()
    call_kwargs = instance.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.llm.openai_provider'`

- [ ] **Step 3: Write minimal implementation**

```python
# providers/llm/openai_provider.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_provider.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/llm/openai_provider.py tests/test_llm_provider.py
git commit -m "feat: add OpenAI LLM provider implementation"
```

---

### Task 6: Anthropic LLM provider (providers/llm/anthropic_provider.py)

**Files:**
- Create: `providers/llm/anthropic_provider.py`
- Create: `tests/test_anthropic_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_anthropic_provider.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from providers.llm.anthropic_provider import AnthropicLLMProvider


def test_anthropic_llm_provider_complete():
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = '{"title": "Test Story"}'

    with patch("providers.llm.anthropic_provider.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicLLMProvider(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        )
        result = asyncio.run(provider.complete("system prompt", "user prompt"))

    assert result == '{"title": "Test Story"}'
    instance.messages.create.assert_called_once()
    call_kwargs = instance.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"
    assert call_kwargs["system"] == "system prompt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_anthropic_provider.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# providers/llm/anthropic_provider.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_anthropic_provider.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/llm/anthropic_provider.py tests/test_anthropic_provider.py
git commit -m "feat: add Anthropic LLM provider implementation"
```

---

### Task 7: OpenAI Image provider (providers/image/openai_provider.py)

**Files:**
- Create: `providers/image/openai_provider.py`
- Create: `tests/test_image_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_image_provider.py
import asyncio
import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from providers.image.openai_provider import OpenAIImageProvider


def test_openai_image_provider_generate():
    fake_image_bytes = b"\x89PNG\r\n\x1a\n"
    fake_b64 = base64.b64encode(fake_image_bytes).decode()

    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].b64_json = fake_b64

    with patch("providers.image.openai_provider.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.images.generate = AsyncMock(return_value=mock_response)

        provider = OpenAIImageProvider(model="dall-e-3", api_key="test-key")
        result = asyncio.run(provider.generate(
            prompt="a boy in rain",
            style="cinematic",
            width=1920,
            height=1080,
        ))

    assert result == fake_image_bytes
    instance.images.generate.assert_called_once()
    call_kwargs = instance.images.generate.call_args[1]
    assert call_kwargs["model"] == "dall-e-3"
    assert "cinematic" in call_kwargs["prompt"]
    assert call_kwargs["response_format"] == "b64_json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_image_provider.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# providers/image/openai_provider.py
import base64
from openai import AsyncOpenAI


class OpenAIImageProvider:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str, style: str, width: int, height: int) -> bytes:
        size = self._resolve_size(width, height)
        response = await self.client.images.generate(
            model=self.model,
            prompt=f"{style}, {prompt}",
            n=1,
            size=size,
            response_format="b64_json",
        )
        b64_data = response.data[0].b64_json
        return base64.b64decode(b64_data)

    def _resolve_size(self, width: int, height: int) -> str:
        """DALL-E 3 only supports specific sizes. Pick closest."""
        if width >= 1792 or height >= 1792:
            return "1792x1024" if width >= height else "1024x1792"
        return "1024x1024"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_image_provider.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/image/openai_provider.py tests/test_image_provider.py
git commit -m "feat: add OpenAI Image provider (DALL-E) implementation"
```

---

### Task 8: OpenAI Audio provider (providers/audio/openai_provider.py)

**Files:**
- Create: `providers/audio/openai_provider.py`
- Create: `tests/test_audio_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audio_provider.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from providers.audio.openai_provider import OpenAIAudioProvider


def test_openai_audio_provider_synthesize():
    fake_audio = b"\xff\xfb\x90\x00"

    mock_response = MagicMock()
    mock_response.content = fake_audio

    with patch("providers.audio.openai_provider.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.audio.speech.create = AsyncMock(return_value=mock_response)

        provider = OpenAIAudioProvider(model="tts-1-hd", api_key="test-key")
        result = asyncio.run(provider.synthesize("你好世界", "alloy"))

    assert result == fake_audio
    instance.audio.speech.create.assert_called_once()
    call_kwargs = instance.audio.speech.create.call_args[1]
    assert call_kwargs["model"] == "tts-1-hd"
    assert call_kwargs["voice"] == "alloy"
    assert call_kwargs["input"] == "你好世界"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audio_provider.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# providers/audio/openai_provider.py
from openai import AsyncOpenAI


class OpenAIAudioProvider:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def synthesize(self, text: str, voice: str) -> bytes:
        response = await self.client.audio.speech.create(
            model=self.model,
            voice=voice,
            input=text,
            response_format="mp3",
        )
        return response.content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audio_provider.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/audio/openai_provider.py tests/test_audio_provider.py
git commit -m "feat: add OpenAI Audio provider (TTS) implementation"
```

---

### Task 9: Provider factory (providers/__init__.py)

**Files:**
- Modify: `providers/__init__.py`
- Create: `tests/test_provider_factory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provider_factory.py
import os
import pytest
from unittest.mock import patch
from schemas.config import ProviderConfig
from providers import create_llm_provider, create_image_provider, create_audio_provider


def test_create_openai_llm_provider():
    config = ProviderConfig.model_validate({
        "type": "openai", "model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"
    })
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        provider = create_llm_provider(config)
    assert provider.model == "gpt-4o"


def test_create_anthropic_llm_provider():
    config = ProviderConfig.model_validate({
        "type": "anthropic", "model": "claude-sonnet-4-20250514", "api_key_env": "ANTHROPIC_API_KEY"
    })
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        provider = create_llm_provider(config)
    assert provider.model == "claude-sonnet-4-20250514"


def test_create_openai_image_provider():
    config = ProviderConfig.model_validate({
        "type": "openai", "model": "dall-e-3", "api_key_env": "OPENAI_API_KEY"
    })
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        provider = create_image_provider(config)
    assert provider.model == "dall-e-3"


def test_create_openai_audio_provider():
    config = ProviderConfig.model_validate({
        "type": "openai", "model": "tts-1-hd", "api_key_env": "OPENAI_API_KEY"
    })
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        provider = create_audio_provider(config)
    assert provider.model == "tts-1-hd"


def test_unknown_provider_type_raises():
    config = ProviderConfig.model_validate({
        "type": "unknown", "model": "x", "api_key_env": "KEY"
    })
    with patch.dict(os.environ, {"KEY": "test"}):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider(config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_provider_factory.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_llm_provider'`

- [ ] **Step 3: Write minimal implementation**

```python
# providers/__init__.py
import os
from schemas.config import ProviderConfig
from providers.base import LLMProvider, ImageProvider, AudioProvider


def create_llm_provider(config: ProviderConfig) -> LLMProvider:
    api_key = os.environ[config.api_key_env]
    match config.type:
        case "openai":
            from providers.llm.openai_provider import OpenAILLMProvider
            return OpenAILLMProvider(model=config.model, api_key=api_key)
        case "anthropic":
            from providers.llm.anthropic_provider import AnthropicLLMProvider
            return AnthropicLLMProvider(model=config.model, api_key=api_key)
        case _:
            raise ValueError(f"Unknown LLM provider: {config.type}")


def create_image_provider(config: ProviderConfig) -> ImageProvider:
    api_key = os.environ[config.api_key_env]
    match config.type:
        case "openai":
            from providers.image.openai_provider import OpenAIImageProvider
            return OpenAIImageProvider(model=config.model, api_key=api_key)
        case _:
            raise ValueError(f"Unknown image provider: {config.type}")


def create_audio_provider(config: ProviderConfig) -> AudioProvider:
    api_key = os.environ[config.api_key_env]
    match config.type:
        case "openai":
            from providers.audio.openai_provider import OpenAIAudioProvider
            return OpenAIAudioProvider(model=config.model, api_key=api_key)
        case _:
            raise ValueError(f"Unknown audio provider: {config.type}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_provider_factory.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/__init__.py tests/test_provider_factory.py
git commit -m "feat: add provider factory with type-based dispatch"
```

---

## Chunk 3: Agent Base and Core Agents (Polish + Storyboard)

### Task 10: Agent base class (agents/base.py)

**Files:**
- Create: `agents/base.py`
- Create: `tests/test_agent_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_base.py
import asyncio
import json
import os
import pytest
from pydantic import BaseModel
from agents.base import BaseAgent
from schemas.config import AppConfig


class SampleOutput(BaseModel):
    value: str


class SampleAgent(BaseAgent[str, SampleOutput]):
    async def run(self, input_data: str) -> SampleOutput:
        return SampleOutput(value=input_data.upper())


def make_config() -> AppConfig:
    return AppConfig.model_validate({
        "providers": {"llm": {"type": "openai", "model": "m", "api_key_env": "K"}},
    })


def test_agent_save_checkpoint(tmp_path):
    agent = SampleAgent(config=make_config(), output_dir=str(tmp_path))
    data = SampleOutput(value="hello")
    agent.save_checkpoint("test_stage", data)
    path = tmp_path / "test_stage.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["value"] == "hello"


def test_agent_load_checkpoint(tmp_path):
    agent = SampleAgent(config=make_config(), output_dir=str(tmp_path))
    data = SampleOutput(value="world")
    agent.save_checkpoint("test_stage", data)
    restored = agent.load_checkpoint("test_stage", SampleOutput)
    assert restored is not None
    assert restored.value == "world"


def test_agent_load_checkpoint_missing(tmp_path):
    agent = SampleAgent(config=make_config(), output_dir=str(tmp_path))
    result = agent.load_checkpoint("nonexistent", SampleOutput)
    assert result is None


def test_agent_run(tmp_path):
    agent = SampleAgent(config=make_config(), output_dir=str(tmp_path))
    result = asyncio.run(agent.run("hello"))
    assert result.value == "HELLO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.base'`

- [ ] **Step 3: Write minimal implementation**

```python
# agents/base.py
from __future__ import annotations
import os
import json
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_base.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/base.py tests/test_agent_base.py
git commit -m "feat: add generic BaseAgent with checkpoint save/load"
```

---

### Task 11: Polish Agent (agents/polish_agent.py)

**Files:**
- Create: `agents/polish_agent.py`
- Create: `tests/test_polish_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_polish_agent.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock
from agents.polish_agent import PolishAgent
from schemas.story import PolishedStory
from schemas.config import AppConfig


def make_config():
    return AppConfig.model_validate({
        "providers": {"llm": {"type": "openai", "model": "gpt-4o", "api_key_env": "K"}},
    })


def test_polish_agent_run(tmp_path):
    expected_response = json.dumps({
        "title": "雨中的少年",
        "full_text": "在一个宁静的小镇上，" + "故事内容" * 50,
        "summary": "少年在雨中奔跑的故事",
        "tone": "warm",
    })

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=expected_response)

    agent = PolishAgent(config=make_config(), output_dir=str(tmp_path), llm=mock_llm)
    result = asyncio.run(agent.run("小明在雨中奔跑"))

    assert isinstance(result, PolishedStory)
    assert result.title == "雨中的少年"
    assert result.tone == "warm"
    assert result.target_duration_seconds > 0
    mock_llm.complete.assert_called_once()


def test_polish_agent_checkpoint_resume(tmp_path):
    """If checkpoint exists, skip LLM call."""
    existing = PolishedStory(
        title="Cached", full_text="text", summary="s", tone="warm",
        target_duration_seconds=60,
    )
    (tmp_path / "polish.json").write_text(existing.model_dump_json())

    mock_llm = AsyncMock()
    agent = PolishAgent(config=make_config(), output_dir=str(tmp_path), llm=mock_llm)
    result = asyncio.run(agent.run("anything"))

    assert result.title == "Cached"
    mock_llm.complete.assert_not_called()


def test_polish_agent_duration_estimation(tmp_path):
    """Duration = len(full_text) / 250 * 60"""
    text_500_chars = "字" * 500  # 500 chars → 120 seconds
    expected_response = json.dumps({
        "title": "T", "full_text": text_500_chars, "summary": "S", "tone": "warm",
    })

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=expected_response)

    agent = PolishAgent(config=make_config(), output_dir=str(tmp_path), llm=mock_llm)
    result = asyncio.run(agent.run("input"))

    assert result.target_duration_seconds == 120
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_polish_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# agents/polish_agent.py
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
        # Check checkpoint
        cached = self.load_checkpoint("polish", PolishedStory)
        if cached is not None:
            return cached

        # Estimate target words (aim for ~2 min video → 500 chars)
        target_words = max(300, len(input_data) * 5)

        system = SYSTEM_PROMPT.format(target_words=target_words)
        response = await self.llm.complete(system, input_data)

        data = json.loads(response)
        # Compute duration from full_text length
        full_text = data["full_text"]
        duration = int(len(full_text) / 250 * 60)
        data["target_duration_seconds"] = duration

        story = PolishedStory.model_validate(data)
        self.save_checkpoint("polish", story)
        return story
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_polish_agent.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/polish_agent.py tests/test_polish_agent.py
git commit -m "feat: add Polish Agent with LLM-driven story expansion"
```

---

### Task 12: Storyboard Agent (agents/storyboard_agent.py)

**Files:**
- Create: `agents/storyboard_agent.py`
- Create: `tests/test_storyboard_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storyboard_agent.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock
from agents.storyboard_agent import StoryboardAgent
from schemas.story import PolishedStory
from schemas.storyboard import Storyboard
from schemas.config import AppConfig


def make_config():
    return AppConfig.model_validate({
        "providers": {"llm": {"type": "openai", "model": "gpt-4o", "api_key_env": "K"}},
    })


def make_story():
    return PolishedStory(
        title="雨中的少年",
        full_text="在一个宁静的小镇上..." + "故事" * 100,
        summary="少年在雨中奔跑",
        tone="warm",
        target_duration_seconds=96,
    )


def test_storyboard_agent_run(tmp_path):
    llm_response = json.dumps({
        "title": "雨中的少年",
        "global_style": "cinematic realism, soft lighting",
        "frames": [
            {
                "frame_id": 1,
                "scene_description": "A quiet small town with cobblestone streets under grey sky",
                "narration_text": "在一个宁静的小镇上",
                "duration_seconds": 8.0,
                "visual_style": "cinematic",
                "transition": "fade",
            },
            {
                "frame_id": 2,
                "scene_description": "A boy starts running in the rain",
                "narration_text": "少年开始在雨中奔跑",
                "duration_seconds": 10.0,
                "visual_style": "cinematic",
                "transition": "dissolve",
            },
        ],
    })

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=llm_response)

    agent = StoryboardAgent(config=make_config(), output_dir=str(tmp_path), llm=mock_llm)
    result = asyncio.run(agent.run(make_story()))

    assert isinstance(result, Storyboard)
    assert len(result.frames) == 2
    assert result.global_style == "cinematic realism, soft lighting"
    assert result.total_duration_seconds == 18.0
    assert result.frames[0].scene_description.startswith("A quiet")


def test_storyboard_agent_checkpoint_resume(tmp_path):
    frames_data = [{
        "frame_id": 1, "scene_description": "s", "narration_text": "n",
        "duration_seconds": 8.0, "visual_style": "v", "transition": "fade",
    }]
    existing = Storyboard(
        title="Cached", global_style="g", frames=frames_data,
        total_duration_seconds=8.0,
    )
    (tmp_path / "storyboard.json").write_text(existing.model_dump_json())

    mock_llm = AsyncMock()
    agent = StoryboardAgent(config=make_config(), output_dir=str(tmp_path), llm=mock_llm)
    result = asyncio.run(agent.run(make_story()))

    assert result.title == "Cached"
    mock_llm.complete.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_storyboard_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# agents/storyboard_agent.py
import json
from agents.base import BaseAgent
from schemas.story import PolishedStory
from schemas.storyboard import Storyboard
from schemas.config import AppConfig
from providers.base import LLMProvider

SYSTEM_PROMPT = """你是一位专业的分镜师。请将以下故事分解为视频分镜脚本。
要求：
1. 将故事分为 {n} 个场景帧
2. 每帧的 scene_description 用英文描述画面（供 AI 绘图），包含具体的人物、环境、光线、构图、情绪
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
        # Compute total_duration from frames
        frames = data.get("frames", [])
        total = sum(f["duration_seconds"] for f in frames)
        data["total_duration_seconds"] = total

        storyboard = Storyboard.model_validate(data)
        self.save_checkpoint("storyboard", storyboard)
        return storyboard
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_storyboard_agent.py -v`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/storyboard_agent.py tests/test_storyboard_agent.py
git commit -m "feat: add Storyboard Agent with frame decomposition"
```

---

## Chunk 4: Image Agent, Audio Agent, and Compose Agent

### Task 13: Image Agent (agents/image_agent.py)

**Files:**
- Create: `agents/image_agent.py`
- Create: `tests/test_image_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_image_agent.py
import asyncio
import os
import pytest
from unittest.mock import AsyncMock
from agents.image_agent import ImageAgent
from schemas.storyboard import Frame, Storyboard
from schemas.media import ImageAsset
from schemas.config import AppConfig


def make_config():
    return AppConfig.model_validate({
        "providers": {"image": {"type": "openai", "model": "dall-e-3", "api_key_env": "K"}},
        "pipeline": {"max_concurrency": 2, "retry_attempts": 1, "request_timeout": 60},
        "output": {"resolution": [1920, 1080]},
    })


def make_storyboard():
    frames = [
        Frame(frame_id=1, scene_description="A boy in rain",
              narration_text="旁白1", duration_seconds=8.0,
              visual_style="cinematic", transition="fade"),
        Frame(frame_id=2, scene_description="A rainbow appears",
              narration_text="旁白2", duration_seconds=10.0,
              visual_style="cinematic", transition="dissolve"),
    ]
    return Storyboard(title="T", global_style="cinematic realism",
                      frames=frames, total_duration_seconds=18.0)


def test_image_agent_generates_all_frames(tmp_path):
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    mock_image = AsyncMock()
    mock_image.generate = AsyncMock(return_value=fake_png)

    agent = ImageAgent(config=make_config(), output_dir=str(tmp_path), image_provider=mock_image)
    result = asyncio.run(agent.run(make_storyboard()))

    assert len(result) == 2
    assert all(isinstance(r, ImageAsset) for r in result)
    assert os.path.exists(result[0].file_path)
    assert os.path.exists(result[1].file_path)
    assert mock_image.generate.call_count == 2


def test_image_agent_skips_existing_files(tmp_path):
    """If image file already exists, skip generation (resume support)."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame_1.png").write_bytes(b"\x89PNG")

    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    mock_image = AsyncMock()
    mock_image.generate = AsyncMock(return_value=fake_png)

    agent = ImageAgent(config=make_config(), output_dir=str(tmp_path), image_provider=mock_image)
    result = asyncio.run(agent.run(make_storyboard()))

    assert len(result) == 2
    # Only frame_2 should have been generated
    assert mock_image.generate.call_count == 1


def test_image_agent_prompt_includes_global_style(tmp_path):
    fake_png = b"\x89PNG"
    mock_image = AsyncMock()
    mock_image.generate = AsyncMock(return_value=fake_png)

    agent = ImageAgent(config=make_config(), output_dir=str(tmp_path), image_provider=mock_image)
    asyncio.run(agent.run(make_storyboard()))

    call_args = mock_image.generate.call_args_list[0]
    prompt = call_args[1]["prompt"] if "prompt" in call_args[1] else call_args[0][0]
    assert "cinematic realism" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_image_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# agents/image_agent.py
import asyncio
import os
from agents.base import BaseAgent
from schemas.storyboard import Storyboard
from schemas.media import ImageAsset
from schemas.config import AppConfig
from providers.base import ImageProvider


class ImageAgent(BaseAgent[Storyboard, list[ImageAsset]]):
    def __init__(self, config: AppConfig, output_dir: str, image_provider: ImageProvider):
        super().__init__(config, output_dir)
        self.image_provider = image_provider
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    async def run(self, input_data: Storyboard) -> list[ImageAsset]:
        semaphore = asyncio.Semaphore(self.config.pipeline.max_concurrency)
        width, height = self.config.output.resolution

        async def generate_frame(frame) -> ImageAsset:
            file_path = os.path.join(self.images_dir, f"frame_{frame.frame_id}.png")

            if os.path.exists(file_path):
                return ImageAsset(
                    frame_id=frame.frame_id,
                    file_path=os.path.abspath(file_path),
                    width=width, height=height,
                )

            prompt = f"{input_data.global_style} style, {frame.scene_description}, {frame.visual_style}"

            async with semaphore:
                image_bytes = await self._generate_with_retry(prompt, frame.visual_style, width, height)

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            return ImageAsset(
                frame_id=frame.frame_id,
                file_path=os.path.abspath(file_path),
                width=width, height=height,
            )

        tasks = [generate_frame(frame) for frame in input_data.frames]
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda r: r.frame_id)

    async def _generate_with_retry(self, prompt: str, style: str, width: int, height: int) -> bytes:
        max_retries = self.config.pipeline.retry_attempts
        for attempt in range(max_retries):
            try:
                return await self.image_provider.generate(
                    prompt=prompt, style=style, width=width, height=height,
                )
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_image_agent.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/image_agent.py tests/test_image_agent.py
git commit -m "feat: add Image Agent with concurrent generation and resume"
```

---

### Task 14: Audio Agent (agents/audio_agent.py)

**Files:**
- Create: `agents/audio_agent.py`
- Create: `tests/test_audio_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audio_agent.py
import asyncio
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.audio_agent import AudioAgent
from schemas.storyboard import Frame, Storyboard
from schemas.media import AudioAsset
from schemas.config import AppConfig


def make_config():
    return AppConfig.model_validate({
        "providers": {"audio": {"type": "openai", "model": "tts-1-hd", "api_key_env": "K", "voice": "alloy"}},
        "pipeline": {"max_concurrency": 2, "retry_attempts": 1, "request_timeout": 60},
    })


def make_storyboard():
    frames = [
        Frame(frame_id=1, scene_description="s1", narration_text="你好世界",
              duration_seconds=8.0, visual_style="v", transition="fade"),
        Frame(frame_id=2, scene_description="s2", narration_text="再见",
              duration_seconds=6.0, visual_style="v", transition="cut"),
    ]
    return Storyboard(title="T", global_style="g", frames=frames, total_duration_seconds=14.0)


def test_audio_agent_generates_all_frames(tmp_path):
    fake_mp3 = b"\xff\xfb\x90\x00" * 100

    mock_audio = AsyncMock()
    mock_audio.synthesize = AsyncMock(return_value=fake_mp3)

    with patch("agents.audio_agent.get_audio_duration", return_value=8.5):
        agent = AudioAgent(config=make_config(), output_dir=str(tmp_path), audio_provider=mock_audio)
        result = asyncio.run(agent.run(make_storyboard()))

    assert len(result) == 2
    assert all(isinstance(r, AudioAsset) for r in result)
    assert result[0].duration_seconds == 8.5
    assert os.path.exists(result[0].file_path)


def test_audio_agent_skips_existing(tmp_path):
    audios_dir = tmp_path / "audios"
    audios_dir.mkdir()
    (audios_dir / "frame_1.mp3").write_bytes(b"\xff\xfb")

    fake_mp3 = b"\xff\xfb\x90\x00" * 100
    mock_audio = AsyncMock()
    mock_audio.synthesize = AsyncMock(return_value=fake_mp3)

    with patch("agents.audio_agent.get_audio_duration", return_value=6.0):
        agent = AudioAgent(config=make_config(), output_dir=str(tmp_path), audio_provider=mock_audio)
        result = asyncio.run(agent.run(make_storyboard()))

    assert len(result) == 2
    assert mock_audio.synthesize.call_count == 1  # Only frame_2


def test_audio_agent_uses_voice_from_config(tmp_path):
    fake_mp3 = b"\xff\xfb"
    mock_audio = AsyncMock()
    mock_audio.synthesize = AsyncMock(return_value=fake_mp3)

    with patch("agents.audio_agent.get_audio_duration", return_value=5.0):
        agent = AudioAgent(config=make_config(), output_dir=str(tmp_path), audio_provider=mock_audio)
        asyncio.run(agent.run(make_storyboard()))

    call_kwargs = mock_audio.synthesize.call_args_list[0][1]
    assert call_kwargs["voice"] == "alloy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audio_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# agents/audio_agent.py
import asyncio
import os
from pydub import AudioSegment
from agents.base import BaseAgent
from schemas.storyboard import Storyboard
from schemas.media import AudioAsset
from schemas.config import AppConfig
from providers.base import AudioProvider


def get_audio_duration(file_path: str) -> float:
    """Get duration of an audio file in seconds using pydub."""
    audio = AudioSegment.from_file(file_path)
    return audio.duration_seconds


class AudioAgent(BaseAgent[Storyboard, list[AudioAsset]]):
    def __init__(self, config: AppConfig, output_dir: str, audio_provider: AudioProvider):
        super().__init__(config, output_dir)
        self.audio_provider = audio_provider
        self.audios_dir = os.path.join(output_dir, "audios")
        os.makedirs(self.audios_dir, exist_ok=True)

    async def run(self, input_data: Storyboard) -> list[AudioAsset]:
        semaphore = asyncio.Semaphore(self.config.pipeline.max_concurrency)
        voice = self.config.providers.get("audio", None)
        voice_name = voice.extra.get("voice", "alloy") if voice else "alloy"

        async def generate_frame(frame) -> AudioAsset:
            file_path = os.path.join(self.audios_dir, f"frame_{frame.frame_id}.mp3")

            if os.path.exists(file_path):
                duration = get_audio_duration(file_path)
                return AudioAsset(
                    frame_id=frame.frame_id,
                    file_path=os.path.abspath(file_path),
                    duration_seconds=duration,
                )

            async with semaphore:
                audio_bytes = await self._synthesize_with_retry(
                    frame.narration_text, voice_name,
                )

            with open(file_path, "wb") as f:
                f.write(audio_bytes)

            duration = get_audio_duration(file_path)
            return AudioAsset(
                frame_id=frame.frame_id,
                file_path=os.path.abspath(file_path),
                duration_seconds=duration,
            )

        tasks = [generate_frame(frame) for frame in input_data.frames]
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda r: r.frame_id)

    async def _synthesize_with_retry(self, text: str, voice: str) -> bytes:
        max_retries = self.config.pipeline.retry_attempts
        for attempt in range(max_retries):
            try:
                return await self.audio_provider.synthesize(text=text, voice=voice)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audio_agent.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/audio_agent.py tests/test_audio_agent.py
git commit -m "feat: add Audio Agent with TTS synthesis and duration detection"
```

---

### Task 15: Compose Agent (agents/compose_agent.py)

**Files:**
- Create: `agents/compose_agent.py`
- Create: `tests/test_compose_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_agent.py
import asyncio
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from agents.compose_agent import ComposeAgent
from schemas.storyboard import Frame, Storyboard
from schemas.media import ImageAsset, AudioAsset, MediaBundle
from schemas.project import VideoProject
from schemas.config import AppConfig


def make_config():
    return AppConfig.model_validate({
        "providers": {},
        "output": {"dir": "./output", "resolution": [1920, 1080], "fps": 30},
    })


def make_project(tmp_path):
    frames = [Frame(frame_id=1, scene_description="s", narration_text="n",
                    duration_seconds=8.0, visual_style="v", transition="fade")]
    sb = Storyboard(title="T", global_style="g", frames=frames, total_duration_seconds=8.0)
    images = [ImageAsset(frame_id=1, file_path=str(tmp_path / "img.png"), width=1920, height=1080)]
    audios = [AudioAsset(frame_id=1, file_path=str(tmp_path / "aud.mp3"), duration_seconds=8.0)]
    return VideoProject(
        storyboard=sb,
        media=MediaBundle(images=images, audios=audios),
        output_path=str(tmp_path / "final.mp4"),
        resolution=[1920, 1080],
        fps=30,
    )


def test_compose_agent_writes_input_props(tmp_path):
    project = make_project(tmp_path)
    remotion_dir = tmp_path / "remotion"
    remotion_dir.mkdir()

    def fake_render(*args, **kwargs):
        # Simulate Remotion creating the output file
        (tmp_path / "final.mp4").write_bytes(b"\x00")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Rendered successfully"
        mock_result.stderr = ""
        return mock_result

    with patch("agents.compose_agent.subprocess.run", side_effect=fake_render):
        agent = ComposeAgent(
            config=make_config(), output_dir=str(tmp_path),
            remotion_dir=str(remotion_dir),
        )
        result = asyncio.run(agent.run(project))

    # Verify input-props.json was written
    props_path = remotion_dir / "input-props.json"
    assert props_path.exists()
    props = json.loads(props_path.read_text())
    assert props["fps"] == 30
    assert props["width"] == 1920
    assert len(props["frames"]) == 1
    assert props["frames"][0]["frameId"] == 1


def test_compose_agent_calls_remotion_cli(tmp_path):
    project = make_project(tmp_path)
    remotion_dir = tmp_path / "remotion"
    remotion_dir.mkdir()

    def fake_render(*args, **kwargs):
        (tmp_path / "final.mp4").write_bytes(b"\x00")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        return mock_result

    with patch("agents.compose_agent.subprocess.run", side_effect=fake_render):
        agent = ComposeAgent(
            config=make_config(), output_dir=str(tmp_path),
            remotion_dir=str(remotion_dir),
        )
        result = asyncio.run(agent.run(project))

    assert result == str(tmp_path / "final.mp4")


def test_compose_agent_raises_on_render_failure(tmp_path):
    project = make_project(tmp_path)
    remotion_dir = tmp_path / "remotion"
    remotion_dir.mkdir()

    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = ""
    mock_process.stderr = "Error: Composition not found"

    with patch("agents.compose_agent.subprocess.run", return_value=mock_process):
        agent = ComposeAgent(
            config=make_config(), output_dir=str(tmp_path),
            remotion_dir=str(remotion_dir),
        )
        with pytest.raises(RuntimeError, match="Remotion render failed"):
            asyncio.run(agent.run(project))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compose_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# agents/compose_agent.py
import json
import os
import subprocess
from agents.base import BaseAgent
from schemas.project import VideoProject
from schemas.config import AppConfig


class ComposeAgent(BaseAgent[VideoProject, str]):
    def __init__(self, config: AppConfig, output_dir: str, remotion_dir: str):
        super().__init__(config, output_dir)
        self.remotion_dir = remotion_dir

    async def run(self, input_data: VideoProject) -> str:
        # Check if video already exists
        if os.path.exists(input_data.output_path):
            return input_data.output_path

        # Write input-props.json for Remotion
        props = self._build_props(input_data)
        props_path = os.path.join(self.remotion_dir, "input-props.json")
        with open(props_path, "w", encoding="utf-8") as f:
            json.dump(props, f, ensure_ascii=False, indent=2)

        # Call Remotion CLI
        output_abs = os.path.abspath(input_data.output_path)
        cmd = [
            "npx", "remotion", "render",
            "src/index.ts", "StoryVideo",
            f"--props=./input-props.json",
            f"--output={output_abs}",
            "--codec=h264",
        ]

        result = subprocess.run(
            cmd,
            cwd=self.remotion_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Remotion render failed (exit code {result.returncode}):\n{result.stderr}"
            )

        return input_data.output_path

    def _build_props(self, project: VideoProject) -> dict:
        frames = []
        for frame in project.storyboard.frames:
            # Find matching media
            image = next(
                (img for img in project.media.images if img.frame_id == frame.frame_id),
                None,
            )
            audio = next(
                (aud for aud in project.media.audios if aud.frame_id == frame.frame_id),
                None,
            )
            frames.append({
                "frameId": frame.frame_id,
                "imagePath": image.file_path if image else "",
                "audioPath": audio.file_path if audio else "",
                "durationSeconds": audio.duration_seconds if audio else frame.duration_seconds,
                "narrationText": frame.narration_text,
                "transition": frame.transition,
            })

        return {
            "fps": project.fps,
            "width": project.resolution[0],
            "height": project.resolution[1],
            "frames": frames,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_compose_agent.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/compose_agent.py tests/test_compose_agent.py
git commit -m "feat: add Compose Agent with Remotion CLI integration"
```

---

## Chunk 5: Orchestrator, Remotion, and SKILL.md

### Task 16: Orchestrator (orchestrator.py)

**Files:**
- Create: `orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator.py
import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from orchestrator import run_pipeline
from schemas.config import AppConfig
from schemas.story import PolishedStory
from schemas.storyboard import Storyboard, Frame
from schemas.media import ImageAsset, AudioAsset


def make_config(tmp_path):
    return AppConfig.model_validate({
        "providers": {
            "llm": {"type": "openai", "model": "gpt-4o", "api_key_env": "K"},
            "image": {"type": "openai", "model": "dall-e-3", "api_key_env": "K"},
            "audio": {"type": "openai", "model": "tts-1-hd", "api_key_env": "K", "voice": "alloy"},
        },
        "pipeline": {"max_concurrency": 2, "retry_attempts": 1, "request_timeout": 60},
        "output": {"dir": str(tmp_path / "output"), "resolution": [1920, 1080], "fps": 30},
    })


def test_run_pipeline_full(tmp_path):
    config = make_config(tmp_path)
    output_dir = str(tmp_path / "output")
    remotion_dir = str(tmp_path / "remotion")
    os.makedirs(remotion_dir, exist_ok=True)

    # Mock all providers
    mock_llm = AsyncMock()
    mock_image = AsyncMock()
    mock_audio = AsyncMock()

    # Polish response
    mock_llm.complete = AsyncMock(side_effect=[
        json.dumps({
            "title": "T", "full_text": "字" * 250,
            "summary": "S", "tone": "warm",
        }),
        json.dumps({
            "title": "T", "global_style": "cinematic",
            "frames": [{
                "frame_id": 1, "scene_description": "scene",
                "narration_text": "旁白", "duration_seconds": 8.0,
                "visual_style": "cinematic", "transition": "fade",
            }],
        }),
    ])

    mock_image.generate = AsyncMock(return_value=b"\x89PNG")
    mock_audio.synthesize = AsyncMock(return_value=b"\xff\xfb" * 50)

    mock_subprocess = MagicMock()
    mock_subprocess.returncode = 0
    mock_subprocess.stdout = ""
    mock_subprocess.stderr = ""

    with patch("orchestrator.create_llm_provider", return_value=mock_llm), \
         patch("orchestrator.create_image_provider", return_value=mock_image), \
         patch("orchestrator.create_audio_provider", return_value=mock_audio), \
         patch("agents.audio_agent.get_audio_duration", return_value=8.5), \
         patch("agents.compose_agent.subprocess.run", return_value=mock_subprocess):

        # Create fake output to simulate Remotion
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, "final.mp4")
        with open(final_path, "wb") as f:
            f.write(b"\x00")

        result = asyncio.run(run_pipeline(
            input_text="小明在雨中奔跑",
            config=config,
            remotion_dir=remotion_dir,
        ))

    assert result == final_path
    assert mock_llm.complete.call_count == 2  # polish + storyboard
    assert mock_image.generate.call_count == 1
    assert mock_audio.synthesize.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# orchestrator.py
import asyncio
import os
import shutil
import sys
import yaml

from schemas.config import AppConfig
from schemas.media import MediaBundle
from schemas.project import VideoProject
from agents.polish_agent import PolishAgent
from agents.storyboard_agent import StoryboardAgent
from agents.image_agent import ImageAgent
from agents.audio_agent import AudioAgent
from agents.compose_agent import ComposeAgent
from providers import create_llm_provider, create_image_provider, create_audio_provider


async def run_pipeline(
    input_text: str,
    config: AppConfig,
    remotion_dir: str = "remotion",
    resume: bool = False,
) -> str:
    output_dir = config.output.dir
    os.makedirs(output_dir, exist_ok=True)

    if not resume and os.path.exists(output_dir):
        # Clean output directory for fresh run
        shutil.rmtree(output_dir)

    # Create providers
    llm = create_llm_provider(config.providers["llm"])
    image_provider = create_image_provider(config.providers["image"])
    audio_provider = create_audio_provider(config.providers["audio"])

    # 1. Polish
    print("[1/5] Polish Agent...", flush=True)
    story = await PolishAgent(config, output_dir, llm=llm).run(input_text)
    print(f"[1/5] Polish Agent... done (title: {story.title})", flush=True)

    # 2. Storyboard
    print("[2/5] Storyboard Agent...", flush=True)
    storyboard = await StoryboardAgent(config, output_dir, llm=llm).run(story)
    print(f"[2/5] Storyboard Agent... done ({len(storyboard.frames)} frames, {storyboard.total_duration_seconds}s)", flush=True)

    # 3. Image + Audio in parallel
    print("[3/5] Image Agent + Audio Agent (parallel)...", flush=True)
    image_agent = ImageAgent(config, output_dir, image_provider=image_provider)
    audio_agent = AudioAgent(config, output_dir, audio_provider=audio_provider)
    images, audios = await asyncio.gather(
        image_agent.run(storyboard),
        audio_agent.run(storyboard),
    )
    print(f"[3/5] Image + Audio done ({len(images)} images, {len(audios)} audios)", flush=True)

    # 4. Compose
    print("[4/5] Compose Agent (Remotion rendering)...", flush=True)
    project = VideoProject(
        storyboard=storyboard,
        media=MediaBundle(images=images, audios=audios),
        output_path=os.path.join(os.path.abspath(output_dir), "final.mp4"),
        resolution=config.output.resolution,
        fps=config.output.fps,
    )
    video_path = await ComposeAgent(config, output_dir, remotion_dir=remotion_dir).run(project)
    print(f"[5/5] Done! -> {video_path}", flush=True)

    return video_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Story Video Pipeline")
    parser.add_argument("text", help="Input text or path to text file")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--remotion-dir", default="remotion", help="Path to Remotion project")
    args = parser.parse_args()

    input_text = open(args.text).read() if os.path.isfile(args.text) else args.text

    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    result = asyncio.run(run_pipeline(
        input_text=input_text,
        config=config,
        remotion_dir=args.remotion_dir,
        resume=args.resume,
    ))
    print(f"\nVideo generated: {result}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add pipeline orchestrator with CLI entry point"
```

---

### Task 17: Remotion project setup

**Files:**
- Create: `remotion/package.json`
- Create: `remotion/tsconfig.json`
- Create: `remotion/src/index.ts`
- Create: `remotion/src/Root.tsx`
- Create: `remotion/src/StoryVideo.tsx`
- Create: `remotion/src/components/SceneFrame.tsx`
- Create: `remotion/src/components/Subtitles.tsx`
- Create: `remotion/src/components/Transition.tsx`

- [ ] **Step 1: Create package.json**

```json
{
  "name": "story-video-remotion",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "preview": "npx remotion preview src/index.ts",
    "render": "npx remotion render src/index.ts StoryVideo"
  },
  "dependencies": {
    "@remotion/cli": "^4.0",
    "remotion": "^4.0",
    "react": "^18",
    "react-dom": "^18"
  },
  "devDependencies": {
    "typescript": "^5.0",
    "@types/react": "^18"
  }
}
```

- [ ] **Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "dist"
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create src/index.ts — Remotion entry**

```typescript
// remotion/src/index.ts
import { registerRoot } from "remotion";
import { Root } from "./Root";

registerRoot(Root);
```

- [ ] **Step 4: Create src/Root.tsx**

```tsx
// remotion/src/Root.tsx
import { Composition } from "remotion";
import { StoryVideo, StoryVideoProps } from "./StoryVideo";

export const Root: React.FC = () => {
  return (
    <Composition
      id="StoryVideo"
      component={StoryVideo}
      // These are overridden by input-props.json at render time
      durationInFrames={300}
      fps={30}
      width={1920}
      height={1080}
      defaultProps={{
        fps: 30,
        width: 1920,
        height: 1080,
        frames: [],
      }}
      calculateMetadata={({ props }) => {
        const totalFrames = props.frames.reduce(
          (sum: number, f: { durationSeconds: number }) => sum + Math.round(f.durationSeconds * props.fps),
          0
        );
        return {
          durationInFrames: Math.max(totalFrames, 1),
          fps: props.fps,
          width: props.width,
          height: props.height,
        };
      }}
    />
  );
};
```

- [ ] **Step 5: Create src/StoryVideo.tsx**

```tsx
// remotion/src/StoryVideo.tsx
import { AbsoluteFill, Audio, Sequence, staticFile } from "remotion";
import { SceneFrame } from "./components/SceneFrame";
import { Subtitles } from "./components/Subtitles";
import { TransitionEffect } from "./components/Transition";

interface FrameData {
  frameId: number;
  imagePath: string;
  audioPath: string;
  durationSeconds: number;
  narrationText: string;
  transition: string;
}

export interface StoryVideoProps {
  fps: number;
  width: number;
  height: number;
  frames: FrameData[];
}

export const StoryVideo: React.FC<StoryVideoProps> = ({ fps, frames }) => {
  const TRANSITION_DURATION = 0.5; // seconds

  let currentFrame = 0;

  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      {frames.map((frame, index) => {
        const durationFrames = Math.round(frame.durationSeconds * fps);
        const startFrame = currentFrame;
        currentFrame += durationFrames;

        return (
          <Sequence
            key={frame.frameId}
            from={startFrame}
            durationInFrames={durationFrames}
          >
            {/* Scene image with Ken Burns */}
            <SceneFrame
              imagePath={frame.imagePath}
              durationFrames={durationFrames}
            />

            {/* Subtitle overlay */}
            <Subtitles
              text={frame.narrationText}
              durationFrames={durationFrames}
            />

            {/* Audio narration */}
            <Audio src={frame.audioPath} />

            {/* Transition effect */}
            {index > 0 && (
              <TransitionEffect
                type={frame.transition}
                durationFrames={Math.round(TRANSITION_DURATION * fps)}
              />
            )}
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
```

- [ ] **Step 6: Create src/components/SceneFrame.tsx**

```tsx
// remotion/src/components/SceneFrame.tsx
import { AbsoluteFill, Img, useCurrentFrame, interpolate } from "remotion";

interface SceneFrameProps {
  imagePath: string;
  durationFrames: number;
}

export const SceneFrame: React.FC<SceneFrameProps> = ({
  imagePath,
  durationFrames,
}) => {
  const frame = useCurrentFrame();

  // Ken Burns: slow zoom from 1.0 to 1.1
  const scale = interpolate(frame, [0, durationFrames], [1.0, 1.1], {
    extrapolateRight: "clamp",
  });

  // Slow pan: drift slightly
  const translateX = interpolate(frame, [0, durationFrames], [0, -20], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <Img
        src={imagePath}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: `scale(${scale}) translateX(${translateX}px)`,
        }}
      />
    </AbsoluteFill>
  );
};
```

- [ ] **Step 7: Create src/components/Subtitles.tsx**

```tsx
// remotion/src/components/Subtitles.tsx
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";

interface SubtitlesProps {
  text: string;
  durationFrames: number;
}

export const Subtitles: React.FC<SubtitlesProps> = ({
  text,
  durationFrames,
}) => {
  const frame = useCurrentFrame();

  // Fade in during first 15 frames
  const opacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        justifyContent: "flex-end",
        alignItems: "center",
        paddingBottom: 60,
      }}
    >
      <div
        style={{
          backgroundColor: "rgba(0, 0, 0, 0.6)",
          padding: "12px 24px",
          borderRadius: 8,
          opacity,
          maxWidth: "80%",
        }}
      >
        <p
          style={{
            color: "white",
            fontSize: 36,
            fontFamily: "sans-serif",
            textAlign: "center",
            margin: 0,
            textShadow: "1px 1px 2px rgba(0,0,0,0.8)",
            lineHeight: 1.4,
          }}
        >
          {text}
        </p>
      </div>
    </AbsoluteFill>
  );
};
```

- [ ] **Step 8: Create src/components/Transition.tsx**

```tsx
// remotion/src/components/Transition.tsx
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";

interface TransitionProps {
  type: string; // "fade" | "dissolve" | "cut"
  durationFrames: number;
}

export const TransitionEffect: React.FC<TransitionProps> = ({
  type,
  durationFrames,
}) => {
  const frame = useCurrentFrame();

  if (type === "cut") return null;

  // For fade and dissolve: overlay a black layer that fades out
  const opacity = interpolate(frame, [0, durationFrames], [1, 0], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: type === "fade" ? "black" : "transparent",
        opacity: type === "fade" ? opacity : 0,
      }}
    />
  );
};
```

- [ ] **Step 9: Install npm dependencies**

Run: `cd remotion && npm install`
Expected: Dependencies install successfully.

- [ ] **Step 10: Type check TypeScript**

Run: `cd remotion && npx tsc --noEmit`
Expected: No type errors. If errors occur, fix the component files and re-run.

- [ ] **Step 11: Commit**

```bash
git add remotion/
git commit -m "feat: add Remotion project with StoryVideo composition and components"
```

---

### Task 18: SKILL.md

**Files:**
- Create: `SKILL.md`

- [ ] **Step 1: Write SKILL.md**

```markdown
---
name: story-video
description: Use when the user wants to generate a video from text, a short story, or a script. Converts text input into a complete video with images, narration audio, and transitions using a multi-agent pipeline.
---

# Story Video

Convert text into narrated video using a 5-agent pipeline: polish, storyboard, image generation, audio synthesis, and video composition.

## Prerequisites

- Python 3.11+
- Node.js 18+
- Chrome/Chromium (required by Remotion for rendering)
- ffmpeg (required by pydub for audio duration detection)
- API keys for configured providers (set as environment variables)

## Quick Start

1. Configure `config.yaml` — set provider types and API key env var names
2. Set environment variables for API keys
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   cd remotion && npm install && cd ..
   ```
4. Run:
   ```bash
   python orchestrator.py "你的故事文字"
   ```

## Pipeline

```
Input text → [Polish] → [Storyboard] → [Image ‖ Audio] → [Compose] → MP4
```

1. **Polish Agent** — Expands short text into a complete story using LLM
2. **Storyboard Agent** — Decomposes story into scene frames with visual descriptions and narration
3. **Image Agent** — Generates an image for each frame (parallel)
4. **Audio Agent** — Synthesizes narration audio for each frame (parallel)
5. **Compose Agent** — Assembles images + audio into video via Remotion

## Configuration

See `config.yaml` for all options. Key settings:

- `providers.llm.type` — LLM provider: `openai` or `anthropic`
- `providers.image.type` — Image provider: `openai` (DALL-E)
- `providers.audio.type` — Audio provider: `openai` (TTS)
- `pipeline.max_concurrency` — Max parallel API requests (default: 3)
- `output.resolution` — Video resolution (default: [1920, 1080])

## Output

```
output/
├── polish.json           # Polished story
├── storyboard.json       # Storyboard with frames
├── images/
│   ├── frame_1.png       # Generated images
│   └── ...
├── audios/
│   ├── frame_1.mp3       # Generated audio
│   └── ...
└── final.mp4             # Final video
```

## Resume

If the pipeline fails mid-way, re-run with `--resume` to continue from the last checkpoint:

```bash
python orchestrator.py "your text" --resume
```

## Extending

Add new providers by implementing the Protocol interface in `providers/base.py` and registering in the factory (`providers/__init__.py`).
```

- [ ] **Step 2: Commit**

```bash
git add SKILL.md
git commit -m "feat: add SKILL.md for OpenCode skill entry point"
```

---

### Task 19: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 2: Fix any failures**

If any tests fail, fix the issues and re-run.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: verify all tests pass"
```
