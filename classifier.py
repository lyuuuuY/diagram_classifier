from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode


MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.pth"
VALID_LABELS = frozenset({"lewis", "graph", "equation"})
REQUIRED_IMAGE_SIZE = (224, 224)
EIGHT_BIT_IMAGE_MODES = frozenset(
    {
        "1",
        "L",
        "LA",
        "P",
        "RGB",
        "RGBA",
        "CMYK",
    }
)

_MODEL: nn.Module | None = None
_TRANSFORM: transforms.Compose | None = None
_CLASS_NAMES: tuple[str, ...] | None = None
_DEVICE: torch.device | None = None
_LOAD_LOCK = threading.Lock()


def _load_checkpoint(
    model_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load the local trusted checkpoint with PyTorch version compatibility."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )
    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    required_fields = {
        "architecture",
        "model_state_dict",
        "class_names",
        "input_size",
        "normalization",
        "dropout",
    }
    missing_fields = required_fields - set(checkpoint)
    if missing_fields:
        raise RuntimeError(
            "Model checkpoint is missing fields: "
            f"{sorted(missing_fields)}"
        )
    if checkpoint["architecture"] != "resnet18":
        raise RuntimeError(
            "Unsupported model architecture: "
            f"{checkpoint['architecture']}"
        )
    return checkpoint


def _initialize() -> None:
    global _MODEL, _TRANSFORM, _CLASS_NAMES, _DEVICE

    if _MODEL is not None:
        return
    with _LOAD_LOCK:
        if _MODEL is not None:
            return

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = _load_checkpoint(MODEL_PATH, device)
        class_names = tuple(checkpoint["class_names"])
        if set(class_names) != VALID_LABELS or len(class_names) != 3:
            raise RuntimeError(
                f"Unexpected checkpoint classes: {class_names}"
            )
        if int(checkpoint["input_size"]) != REQUIRED_IMAGE_SIZE[0]:
            raise RuntimeError(
                "The checkpoint input size is not 224"
            )

        model = resnet18(weights=None)
        input_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=float(checkpoint["dropout"])),
            nn.Linear(input_features, len(class_names)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        normalization = checkpoint["normalization"]
        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    normalization["mean"],
                    normalization["std"],
                ),
            ]
        )

        _DEVICE = device
        _CLASS_NAMES = class_names
        _TRANSFORM = image_transform
        _MODEL = model


def _read_png(filepath: str) -> Image.Image:
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")

    image_path = Path(filepath)
    if not image_path.is_file():
        raise FileNotFoundError(f"PNG file not found: {image_path}")

    with Image.open(image_path) as source_image:
        if source_image.format != "PNG":
            raise ValueError("Input file must be a PNG image")
        if source_image.size != REQUIRED_IMAGE_SIZE:
            raise ValueError(
                "Input PNG must be exactly 224x224 pixels; "
                f"received {source_image.width}x{source_image.height}"
            )
        if source_image.mode not in EIGHT_BIT_IMAGE_MODES:
            raise ValueError(
                "Input PNG must use 8-bit color channels in the "
                f"0-255 range; received mode {source_image.mode!r}"
            )
        return source_image.convert("RGB")


def classify(filepath: str) -> str:
    """Classify one 224x224 PNG as 'lewis', 'graph', or 'equation'."""
    image = _read_png(filepath)
    _initialize()

    input_tensor = _TRANSFORM(image).unsqueeze(0).to(_DEVICE)
    with torch.inference_mode():
        logits = _MODEL(input_tensor)
        predicted_index = int(logits.argmax(dim=1).item())

    result = _CLASS_NAMES[predicted_index]
    return result


__all__ = ["classify"]
