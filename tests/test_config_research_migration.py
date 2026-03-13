import json

from nanobot.config.loader import load_config


def test_load_config_migrates_legacy_research_backend(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"research": {"defaultBackend": "native"}}),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.research.default_backend == "opencode"
