import os
import shutil
import whisper
from pathlib import Path

HOME_DIR = os.path.expanduser("~")
MODEL_CACHE_DIR = Path(HOME_DIR) / ".cache" / "whisper"
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]

# Nome real do arquivo no cache do whisper por modelo
MODEL_FILENAMES = {
    "tiny":   "tiny.pt",
    "base":   "base.pt",
    "small":  "small.pt",
    "medium": "medium.pt",
    "large":  "large.pt",
    "turbo":  "large-v3-turbo.pt",
}

# Tamanhos aproximados em MB
MODEL_SIZES_MB = {
    "tiny":   75,
    "base":   145,
    "small":  465,
    "medium": 1500,
    "large":  2900,
    "turbo":  1600,
}


def get_model_path(model_name: str) -> Path:
    filename = MODEL_FILENAMES.get(model_name, f"{model_name}.pt")
    return MODEL_CACHE_DIR / filename


def is_model_downloaded(model_name: str) -> bool:
    return get_model_path(model_name).exists()


def get_model_size_on_disk(model_name: str) -> int:
    path = get_model_path(model_name)
    return path.stat().st_size if path.exists() else 0


def get_downloaded_models() -> list:
    return [m for m in AVAILABLE_MODELS if is_model_downloaded(m)]


def download_model(model_name: str):
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    whisper.load_model(model_name, device="cpu")


def delete_model(model_name: str):
    path = get_model_path(model_name)
    if path.exists():
        path.unlink()


def load_model_with_check(model_name: str, device: str = "cpu"):
    return whisper.load_model(model_name, device=device)


def get_model_info() -> dict:
    downloaded = get_downloaded_models()
    return {
        "available": AVAILABLE_MODELS,
        "downloaded": downloaded,
        "cache_dir": str(MODEL_CACHE_DIR),
    }
