"""Portable paths — works locally and on Kaggle."""
from pathlib import Path

if Path("/kaggle/input").exists():
    PROJECT_ROOT = Path("/kaggle/working")
    DATA_ROOT = next(Path("/kaggle/input").iterdir())
else:
    _cwd = Path.cwd()
    PROJECT_ROOT = _cwd.parent if _cwd.name == "notebooks" else _cwd
    DATA_ROOT = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    )

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_PATH = DATA_ROOT / "training_data.csv"
RECORDINGS_DIR = DATA_ROOT / "training_data"

RANDOM_SEED = 42
