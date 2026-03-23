# Story Video Skill Design

## Overview

A multi-agent pipeline implemented as an OpenCode Skill that converts short text into a complete narrated video. Five specialized agents collaborate in sequence: polish text, generate storyboard, create images, synthesize audio, and compose the final video.

**Language scope:** This skill is designed primarily for Chinese text input. Narration audio is generated in Chinese. The prompts instruct the LLM to produce Chinese narration text and English scene descriptions (for image generation). Supporting other languages would require prompt and voice configuration changes — out of scope for v1.

## Decisions

| Dimension | Decision |
|-----------|----------|
| Form factor | OpenCode Skill |
| Tech stack | Python (main) + TypeScript (Remotion only) |
| AI services | All pluggable via Protocol interfaces |
| Video composition | Remotion (Node.js), invoked via subprocess |
| Agent orchestration | Linear pipeline with parallel image+audio stage |

## Architecture

### Directory Structure

```
story-video-skill/
├── SKILL.md                        # OpenCode Skill entry point
├── config.yaml                     # Default config (provider selection)
├── requirements.txt                # Python dependencies
├── orchestrator.py                 # Pipeline orchestrator (entry point)
├── schemas/
│   ├── __init__.py
│   ├── story.py                    # PolishedStory
│   ├── storyboard.py               # Storyboard, Frame
│   ├── media.py                    # ImageAsset, AudioAsset, MediaBundle
│   └── project.py                  # VideoProject
├── agents/
│   ├── __init__.py
│   ├── base.py                     # Agent base class
│   ├── polish_agent.py             # Polish Agent
│   ├── storyboard_agent.py         # Storyboard Agent
│   ├── image_agent.py              # Image Agent
│   ├── audio_agent.py              # Audio Agent
│   └── compose_agent.py            # Compose Agent
├── providers/
│   ├── __init__.py
│   ├── base.py                     # Protocol definitions
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── openai_provider.py      # GPT-4o implementation
│   │   └── anthropic_provider.py   # Claude implementation
│   ├── image/
│   │   ├── __init__.py
│   │   └── openai_provider.py      # DALL-E implementation
│   └── audio/
│       ├── __init__.py
│       └── openai_provider.py      # OpenAI TTS implementation
├── remotion/
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── index.ts                # Remotion entry
│   │   ├── Root.tsx                 # Root component
│   │   ├── StoryVideo.tsx           # Main video component
│   │   └── components/
│   │       ├── SceneFrame.tsx       # Single frame: image + Ken Burns
│   │       ├── Subtitles.tsx        # Subtitle overlay
│   │       └── Transition.tsx       # Transition effects
└── output/                         # Default output directory (gitignored)
```

### Pipeline Flow

```
User input text (str)
     │
     ▼
┌──────────────┐
│ Polish Agent  │  short text → full story
└──────┬───────┘
       │ PolishedStory
       ▼
┌──────────────────┐
│ Storyboard Agent  │  story → storyboard (N frames)
└──────┬───────────┘
       │ Storyboard (list[Frame])
       ├────────────────────┐
       ▼                    ▼
┌─────────────┐     ┌──────────────┐
│ Image Agent  │     │ Audio Agent   │   parallel
└──────┬──────┘     └──────┬───────┘
       │ [ImageAsset]      │ [AudioAsset]
       └────────┬──────────┘
                ▼
        ┌───────────────┐
        │ Compose Agent  │  images + audio → Remotion → MP4
        └───────┬───────┘
                │
                ▼
          output/final.mp4
```

### Core Design Principles

1. **Data-driven** — Each agent receives a Pydantic model and outputs a Pydantic model. The orchestrator handles passing data between agents.
2. **Pluggable providers** — LLM/image/audio services defined via Python Protocol. Switched by config.yaml.
3. **Sync orchestration + local parallelism** — Orchestrator calls agents linearly; image and audio stages run in parallel via `asyncio.gather`.
4. **Checkpoint persistence** — Each stage saves output as JSON/files to `output/`, enabling resume from failure.

## Data Schemas

### schemas/story.py — Polish Agent Output

```python
class PolishedStory(BaseModel):
    title: str                    # Story title
    full_text: str                # Polished and expanded full text
    summary: str                  # One-line summary (for global style guidance)
    tone: str                     # Tone: warm/suspense/inspirational/humor...
    target_duration_seconds: int  # Target video duration (estimated from word count)
```

Duration estimation formula: `target_duration_seconds = len(full_text) / 250 * 60` (Chinese ~250 chars/minute speaking rate).

### schemas/storyboard.py — Storyboard Agent Output

```python
class Frame(BaseModel):
    frame_id: int                 # Frame sequence number
    scene_description: str        # Visual description (English, for image generation)
    narration_text: str           # Narration text for this frame (Chinese)
    duration_seconds: float       # Frame duration
    visual_style: str             # Visual style hint (watercolor/cyberpunk/realistic...)
    transition: str               # Transition effect: fade/cut/dissolve

class Storyboard(BaseModel):
    title: str
    global_style: str             # Global visual style (for consistency)
    frames: list[Frame]           # Ordered frame list
    total_duration_seconds: float # Total duration
```

### schemas/media.py — Image/Audio Agent Outputs

```python
class ImageAsset(BaseModel):
    frame_id: int                 # Associated frame sequence number
    file_path: str                # Generated image file path
    width: int
    height: int

class AudioAsset(BaseModel):
    frame_id: int                 # Associated frame sequence number
    file_path: str                # Generated audio file path
    duration_seconds: float       # Actual audio duration

class MediaBundle(BaseModel):
    images: list[ImageAsset]
    audios: list[AudioAsset]
```

### schemas/project.py — Compose Agent Input

```python
class VideoProject(BaseModel):
    """Complete input for the Compose Agent, aggregating all previous stage outputs."""
    storyboard: Storyboard
    media: MediaBundle
    output_path: str              # Final video output path
    resolution: list[int]          # [width, height], default [1920, 1080]
    fps: int                      # Default 30
```

### Data Flow Summary

```
User text (str)
  → PolishedStory
    → Storyboard
      → MediaBundle (images[] + audios[])
        → VideoProject
          → final.mp4
```

`frame_id` is the association key across all stages.

## Provider Layer

### Interface Definitions — providers/base.py

```python
from typing import Protocol

class LLMProvider(Protocol):
    """Used by Polish Agent and Storyboard Agent."""
    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...

class ImageProvider(Protocol):
    """Used by Image Agent."""
    async def generate(
        self, prompt: str, style: str, width: int, height: int
    ) -> bytes: ...

class AudioProvider(Protocol):
    """Used by Audio Agent — speech synthesis."""
    async def synthesize(
        self, text: str, voice: str
    ) -> bytes: ...
```

Design points:
- Minimal interfaces — each provider has exactly 1 core method
- All `async` — ready for parallel execution
- Simple returns — LLM returns string, image/audio return raw bytes; file I/O is the agent's responsibility

### Configuration — config.yaml

```yaml
providers:
  llm:
    type: openai          # openai | anthropic
    model: gpt-4o
    api_key_env: OPENAI_API_KEY

  image:
    type: openai          # openai | (future: flux, midjourney...)
    model: dall-e-3
    api_key_env: OPENAI_API_KEY
    default_size: [1920, 1080]

  audio:
    type: openai          # openai | (future: fish-audio, elevenlabs...)
    model: tts-1-hd
    voice: alloy
    api_key_env: OPENAI_API_KEY

pipeline:
  max_concurrency: 3    # Max parallel requests for image/audio generation
  retry_attempts: 3     # Retries per frame on failure
  request_timeout: 60   # Seconds per provider API call

output:
  dir: ./output
  resolution: [1920, 1080]
  fps: 30
```

### Provider Factory — providers/__init__.py

```python
def create_llm_provider(config: dict) -> LLMProvider:
    match config["type"]:
        case "openai":
            return OpenAILLMProvider(config)
        case "anthropic":
            return AnthropicLLMProvider(config)
        case _:
            raise ValueError(f"Unknown LLM provider: {config['type']}")

# create_image_provider, create_audio_provider follow same pattern
```

### Built-in Implementations (First Batch)

| Provider | Type | File |
|----------|------|------|
| OpenAI GPT-4o | LLM | `providers/llm/openai_provider.py` |
| Anthropic Claude | LLM | `providers/llm/anthropic_provider.py` |
| OpenAI DALL-E 3 | Image | `providers/image/openai_provider.py` |
| OpenAI TTS | Audio | `providers/audio/openai_provider.py` |

### Adding New Providers

1. Implement the corresponding Protocol interface
2. Register the `type` name in the factory function
3. Switch `type` in `config.yaml`

No agent code changes required.

## Agent Details

### Agent Base Class — agents/base.py

```python
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

class BaseAgent(ABC, Generic[TIn, TOut]):
    def __init__(self, config: AppConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir

    @abstractmethod
    async def run(self, input_data: TIn) -> TOut: ...

    def save_checkpoint(self, stage: str, data: BaseModel):
        """Save intermediate output to output/{stage}.json for resume support."""
        path = os.path.join(self.output_dir, f"{stage}.json")
        with open(path, "w") as f:
            f.write(data.model_dump_json(indent=2))

    def load_checkpoint(self, stage: str, model_cls):
        """Try to load existing checkpoint; return None if not found."""
        path = os.path.join(self.output_dir, f"{stage}.json")
        if os.path.exists(path):
            return model_cls.model_validate_json(open(path).read())
        return None
```

Each agent parameterizes `TIn` and `TOut`:
- `PolishAgent(BaseAgent[str, PolishedStory])`
- `StoryboardAgent(BaseAgent[PolishedStory, Storyboard])`
- `ImageAgent(BaseAgent[Storyboard, list[ImageAsset]])`
- `AudioAgent(BaseAgent[Storyboard, list[AudioAsset]])`
- `ComposeAgent(BaseAgent[VideoProject, str])`

### 1. Polish Agent — polish_agent.py

**Responsibility:** Short text → complete story

| Item | Detail |
|------|--------|
| Input | `str` (user's raw text) |
| Output | `PolishedStory` |
| Provider | `LLMProvider` |
| Checkpoint | `output/polish.json` |

Core logic:
- Construct system prompt: instruct LLM to act as professional screenwriter, expand short text into a complete story with narrative arc
- Constraints: preserve original meaning, control length (estimate word count from target duration, ~250 Chinese chars/minute)
- Require structured JSON output, parse directly to `PolishedStory`

Prompt strategy:
```
You are a professional story screenwriter. Please polish and expand the following short text into a complete story.
Requirements:
1. Preserve the core theme and emotion of the original
2. Add vivid details, dialogue, and scene descriptions
3. Target approximately {target_words} words
4. Clear story structure: beginning/development/climax/ending
5. Return JSON format: {title, full_text, summary, tone}
```

### 2. Storyboard Agent — storyboard_agent.py

**Responsibility:** Complete story → storyboard script

| Item | Detail |
|------|--------|
| Input | `PolishedStory` |
| Output | `Storyboard` |
| Provider | `LLMProvider` |
| Checkpoint | `output/storyboard.json` |

Core logic:
- Split story into N scene frames (Frame)
- Each frame contains: scene description (English, for image generation), narration text (Chinese), duration, transition effect
- Define global visual style, ensure consistency across all frames

Prompt strategy:
```
You are a professional storyboard artist. Please decompose the following story into a video storyboard.
Requirements:
1. Split the story into {n} scene frames
2. scene_description in English (for AI image generation)
3. narration_text is the narration for that segment (Chinese)
4. Set a unified global_style for visual consistency
5. Each frame 5-15 seconds, total ~{duration} seconds
6. scene_description must include specific: characters, environment, lighting, composition, mood
7. Return JSON format: {title, global_style, frames: [...]}
```

Frame count estimation: `total_duration / avg_frame_duration(8s)`, approximately 8-15 frames.

### 3. Image Agent — image_agent.py

**Responsibility:** Storyboard frames → corresponding images

| Item | Detail |
|------|--------|
| Input | `Storyboard` |
| Output | `list[ImageAsset]` |
| Provider | `ImageProvider` |
| Checkpoint | Individual image files + `output/images.json` metadata |

Core logic:
- Iterate `storyboard.frames`, call `ImageProvider.generate()` for each
- Prompt construction: `global_style + frame.visual_style + frame.scene_description`
- Concurrent generation via `asyncio.gather` (with configurable concurrency limit)
- Save each image to `output/images/frame_{id}.png`; skip if already exists (resume support)

Style consistency:
- Inject `global_style` as prefix to every frame prompt
- Format: `"{global_style} style, {scene_description}, {visual_style}"`

### 4. Audio Agent — audio_agent.py

**Responsibility:** Storyboard frame narration → voice audio

| Item | Detail |
|------|--------|
| Input | `Storyboard` |
| Output | `list[AudioAsset]` |
| Provider | `AudioProvider` |
| Checkpoint | Individual audio files + `output/audios.json` metadata |

Core logic:
- Iterate `storyboard.frames`, call `AudioProvider.synthesize()` for each frame's `narration_text`
- Concurrent generation (same as Image Agent)
- Save to `output/audios/frame_{id}.mp3`
- Read actual audio duration using `pydub` (chosen over `mutagen` for simpler API and broader format support)
- Update `AudioAsset.duration_seconds` — this value overrides the estimated frame duration from storyboard

### 5. Compose Agent — compose_agent.py

**Responsibility:** Images + audio → final video

| Item | Detail |
|------|--------|
| Input | `VideoProject` |
| Output | `str` (final video file path) |
| Dependency | Remotion CLI |

Core logic:
1. Serialize `VideoProject` to JSON, write to `remotion/input-props.json`
   - All file paths in `input-props.json` are **absolute paths** to avoid ambiguity
2. Call Remotion CLI from the `remotion/` directory as working directory:
   ```bash
   npx remotion render src/index.ts StoryVideo \
     --props=./input-props.json \
     --output={absolute_output_path}/final.mp4 \
     --codec=h264
   ```
3. Wait for render completion, return video path

### Configuration Schema — schemas/config.py

```python
class ProviderConfig(BaseModel):
    type: str
    model: str
    api_key_env: str
    # Provider-specific fields passed through
    extra: dict = {}

class AppConfig(BaseModel):
    providers: dict[str, ProviderConfig]  # keyed by "llm", "image", "audio"
    output: OutputConfig

class OutputConfig(BaseModel):
    dir: str = "./output"
    resolution: list[int] = [1920, 1080]
    fps: int = 30
```

Config is loaded from `config.yaml` and validated as `AppConfig` at startup, before any agent runs.

### Orchestrator — orchestrator.py

```python
async def run_pipeline(input_text: str, config: AppConfig):
    output_dir = config.output.dir

    # 1. Polish
    story = await PolishAgent(config, output_dir).run(input_text)

    # 2. Storyboard
    storyboard = await StoryboardAgent(config, output_dir).run(story)

    # 3. Image + Audio in parallel
    image_agent = ImageAgent(config, output_dir)
    audio_agent = AudioAgent(config, output_dir)
    images, audios = await asyncio.gather(
        image_agent.run(storyboard),
        audio_agent.run(storyboard),
    )

    # 4. Compose
    project = VideoProject(
        storyboard=storyboard,
        media=MediaBundle(images=images, audios=audios),
        output_path=os.path.join(output_dir, "final.mp4"),
        resolution=config.output.resolution,
        fps=config.output.fps,
    )
    video_path = await ComposeAgent(config, output_dir).run(project)

    return video_path
```

### CLI Entry Point

```python
# orchestrator.py __main__ block
if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser(description="Story Video Pipeline")
    parser.add_argument("text", help="Input text or path to text file")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Load text from file if path provided, otherwise use as literal text
    input_text = open(args.text).read() if os.path.isfile(args.text) else args.text

    # Load and validate config
    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    asyncio.run(run_pipeline(input_text, config))
```

## Remotion Integration

### Remotion Project Structure

```
remotion/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts              # Register composition
│   ├── Root.tsx               # Remotion Root
│   ├── StoryVideo.tsx         # Main video component — orchestrates all scene frames
│   └── components/
│       ├── SceneFrame.tsx     # Single frame: image + Ken Burns pan/zoom
│       ├── Subtitles.tsx      # Subtitle overlay layer
│       └── Transition.tsx     # Transition effects (fade/dissolve/cut)
└── input-props.json           # Generated by Python, passed at render time
```

### Data Transfer: Python → Remotion

Compose Agent serializes `VideoProject` into `input-props.json`:

```json
{
  "fps": 30,
  "width": 1920,
  "height": 1080,
  "frames": [
    {
      "frameId": 1,
      "imagePath": "/absolute/path/to/output/images/frame_1.png",
      "audioPath": "/absolute/path/to/output/audios/frame_1.mp3",
      "durationSeconds": 8.2,
      "narrationText": "在一个宁静的小镇上...",
      "transition": "fade"
    }
  ]
}
```

All media file paths are **absolute** to avoid working-directory ambiguity between Python and Remotion.

### Core Components

**StoryVideo.tsx — Main component:**
Arranges all scene frames along the timeline. Each frame's actual duration is determined by `AudioAsset.duration_seconds`. Remotion frame count = `duration_seconds * fps`. Transition overlap ~0.5s for fade/dissolve.

**SceneFrame.tsx — Single frame:**
Displays one image with Ken Burns slow zoom/pan effect. Uses `staticImage()` for local image loading, `useCurrentFrame()` to drive animation. Scale from 1.0 → 1.1 with slow position drift to avoid slideshow feel.

**Subtitles.tsx — Subtitle layer:**
Overlays narration text at bottom of frame. Semi-transparent background bar with white outlined text. Timing synced to audio duration.

**Transition.tsx — Transitions:**
Three modes: cut (instant switch), fade (fade out + fade in), dissolve (cross-dissolve). Implemented with Remotion's `interpolate` + `Sequence`.

### Render Process

```
1. Python writes remotion/input-props.json (all media paths are absolute)
2. subprocess call (cwd=remotion/):
   npx remotion render src/index.ts StoryVideo \
     --props=./input-props.json \
     --output={absolute_path}/output/final.mp4 \
     --codec=h264
3. Remotion reads props → assembles frames on timeline → encodes to MP4
4. Python checks output/final.mp4 exists → returns path
```

Note: `render.ts` is removed from the design — the direct `npx remotion render` CLI invocation is sufficient. No custom render script needed.

### Dependencies

```json
{
  "dependencies": {
    "@remotion/cli": "^4.0",
    "remotion": "^4.0",
    "react": "^18",
    "react-dom": "^18"
  }
}
```

Rendering requires Chrome/Chromium installed (Remotion uses Puppeteer for frame capture).

## SKILL.md Design

### Frontmatter

```yaml
---
name: story-video
description: Use when the user wants to generate a video from text, a short story, or a script. Converts text input into a complete video with images, narration audio, and transitions using a multi-agent pipeline.
---
```

### Content Structure

1. **Overview** — One-line description: multi-agent pipeline that converts text to video
2. **Prerequisites** — Python 3.11+, Node.js 18+, Chrome/Chromium (for Remotion rendering), pydub + ffmpeg (for audio duration), API keys for configured providers
3. **Quick Start** — Configure config.yaml, install dependencies, run orchestrator
4. **Pipeline Stages** — Brief description of 5 agents and execution order
5. **Configuration** — config.yaml field reference
6. **Output** — Output directory structure, intermediate files, final video
7. **Extending** — How to add new providers

### User Flow

```
User: "Generate a video from this text: 小明在雨中奔跑..."

OpenCode loads story-video skill
  ↓
OpenCode agent follows SKILL.md:
  1. Check environment (Python, Node, Chromium, API keys)
  2. Execute: python orchestrator.py "小明在雨中奔跑..."
  3. Monitor pipeline output logs
  4. Return result: "Video generated: output/final.mp4"
```

### Error Handling & Logging

Progress output at each stage:
```
[1/5] Polish Agent... ✓ (title: 雨中的少年)
[2/5] Storyboard Agent... ✓ (12 frames, total 96s)
[3/5] Image Agent + Audio Agent running in parallel...
  [Image] frame 1/12 ✓
  [Audio] frame 1/12 ✓
  ...
  [Image] 12/12 ✓  [Audio] 12/12 ✓
[4/5] Compose Agent... Remotion rendering...
[5/5] Done! → output/final.mp4 (1m36s, 48MB)
```

On failure, checkpoints are saved. Next run with `--resume` auto-resumes from the last successful stage.

## Error Handling

### Per-frame Failure in Parallel Stages (Image/Audio)

Image and audio generation use `asyncio.Semaphore` to limit concurrency (default: 3 concurrent requests, configurable). When a single frame fails:

1. **Retry** — Each frame generation retries up to 3 times with exponential backoff (1s, 2s, 4s)
2. **Fail the stage** — If retries are exhausted for any frame, the entire stage (image or audio) fails
3. **Checkpoint partial progress** — Successfully generated frames are already saved to disk individually. On the next `--resume` run, existing files are skipped

No partial video is generated — both image and audio stages must complete fully before compose begins.

### Provider API Errors

| Error | Handling |
|-------|----------|
| Auth failure (401/403) | Fail immediately with clear message: "API key invalid or missing for {provider}" |
| Rate limit (429) | Retry with exponential backoff, respect `Retry-After` header if present |
| Network error | Retry up to 3 times |
| Timeout | 60s default timeout per request, retry on timeout |
| Invalid response | Fail the frame, enter retry loop |

### Checkpoint Resume Logic

The orchestrator checks for existing checkpoints at each stage start:

```
1. output/polish.json exists?       → skip polish, load PolishedStory
2. output/storyboard.json exists?   → skip storyboard, load Storyboard
3. output/images/ has all frames?   → skip image generation (partial = generate missing only)
4. output/audios/ has all frames?   → skip audio generation (partial = generate missing only)
5. output/final.mp4 exists?         → skip compose, return path
```

Resume is triggered by `--resume` flag. Without it, `output/` is cleared and pipeline runs from scratch.

## Testing Strategy

### Unit Testing (with Mock Providers)

Each agent can be tested independently by injecting mock providers:

- **Mock LLMProvider** — returns predefined JSON strings
- **Mock ImageProvider** — returns a 1x1 pixel PNG
- **Mock AudioProvider** — returns a short silent MP3

Test each agent's input parsing, output schema validation, and checkpoint save/load.

### Integration Testing

Run the full pipeline with mock providers to verify:
- Data flows correctly between stages
- Parallel execution works
- Checkpoint resume works
- Remotion receives valid input-props.json

### Remotion Testing

Use Remotion's `npx remotion preview` for visual verification during development. Test with a small number of frames (2-3) to keep render time short.
