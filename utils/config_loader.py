from __future__ import annotations

from pathlib import Path
import os
import yaml


def resolve_config_path(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path)
    env_path = os.environ.get("PROJECT_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return Path("config") / "config.yaml"


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("runtime", {})["root_dir"] = str(path.parent.parent)
    config.setdefault("runtime", {})["config_path"] = str(path)
    return config


def resolve_path(config: dict, *keys: str) -> Path:
    value = config
    for key in keys:
        value = value[key]
    path = Path(value)
    if not path.is_absolute():
        root = Path(config["runtime"]["root_dir"])
        path = root / path
    return path
