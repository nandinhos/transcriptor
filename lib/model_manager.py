import os
import whisper
from pathlib import Path

HOME_DIR = os.path.expanduser("~")
MODEL_CACHE_DIR = Path(HOME_DIR) / ".cache" / "whisper"
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]


def get_model_path(model_name: str) -> Path:
    return MODEL_CACHE_DIR / f"{model_name}.pt"


def is_model_downloaded(model_name: str) -> bool:
    model_path = get_model_path(model_name)
    if model_path.exists():
        return True
    cache_dirs = [
        Path(HOME_DIR) / ".cache" / "huggingface",
    ]
    for cache_dir in cache_dirs:
        if cache_dir.exists() and os.access(cache_dir, os.R_OK):
            for pattern in [f"{model_name}*", f"*-{model_name}*"]:
                if list(cache_dir.rglob(pattern)):
                    return True
    return False


def get_downloaded_models() -> list:
    downloaded = []
    for model in AVAILABLE_MODELS:
        if is_model_downloaded(model):
            downloaded.append(model)
    return downloaded


def load_model_with_check(model_name: str, device: str = "cuda"):
    if not is_model_downloaded(model_name):
        print(f"Modelo {model_name} não encontrado em cache. Baixando...")

    return whisper.load_model(model_name, device=device)


def get_model_info() -> dict:
    downloaded = get_downloaded_models()
    return {
        "available": AVAILABLE_MODELS,
        "downloaded": downloaded,
        "cache_dir": str(MODEL_CACHE_DIR),
    }
