# smirk/utils/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

def get_config_path(relative: str) -> Path:
    return CONFIGS_DIR / relative