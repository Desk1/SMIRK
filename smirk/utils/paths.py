# smirk/utils/paths.py - path resolving utils

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

def get_config_path(relative: str):
    return str(CONFIGS_DIR / relative)