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
