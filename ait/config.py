"""
Persistent user configuration for AiT.

Stores a small JSON file in the user's home directory so settings like the
SAM3 model checkpoint path survive across sessions. Cross-platform via
pathlib.Path.home() with no external dependencies.

Locations:
    Windows: %USERPROFILE%\\.ait\\config.json
    macOS:   ~/.ait/config.json
    Linux:   ~/.ait/config.json
"""

import json
from pathlib import Path


CONFIG_DIR = Path.home() / ".ait"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    """Read the config file. Returns {} if missing or unreadable."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_config(config: dict) -> None:
    """Write the config file, creating the directory if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_sam3_model_path() -> str | None:
    """Return the saved SAM3 model path, or None if not set."""
    return load_config().get("sam3_model_path")


def set_sam3_model_path(path: str | Path) -> None:
    """Persist the SAM3 model path."""
    config = load_config()
    config["sam3_model_path"] = str(path)
    save_config(config)
