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
2. Set environment variables for API keys:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
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
3. **Image Agent** — Generates an image for each frame (parallel with Audio)
4. **Audio Agent** — Synthesizes narration audio for each frame (parallel with Image)
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
python orchestrator.py "你的文字" --resume
```

## Extending

Add new providers by implementing the Protocol interface in `providers/base.py` and registering in the factory (`providers/__init__.py`).
