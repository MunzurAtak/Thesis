import json
from pathlib import Path


def load_json_config(config_path: str) -> dict:
    """
    Load a json configuration file.

    Args:
        config_path (str): Path to the json configuration file.

    Returns:
        Config dictionary.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
