"""Search for hyperparameters for a three-class diagram classifier.

The script performs:

1. A 16-combination coarse grid search.
2. Re-training of the top configurations with several random seeds.
3. Selection and saving of the best configuration and seed.

This script does not train or save ``best_model.pth`` and does not evaluate the
testing set. After reviewing the saved search results, enter the desired values
in ``train_best_model.py`` and run that script separately.

Expected dataset layout:

    merged_data/
        training/
            equation/
            graph/
            lewis/
        validation/
            equation/
            graph/
            lewis/
        testing/
            equation/
            graph/
            lewis/

Run from the project root:

    .\.venv\Scripts\python.exe parameter_search.py

The ImageNet weights are provided through torchvision. On their first use,
torchvision may download them into the local PyTorch model cache.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "merged_data"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "training_results"

EXPECTED_CLASSES = {"equation", "graph", "lewis"}
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HEAD_MAX_EPOCHS = 10
FINETUNE_MAX_EPOCHS = 20
HEAD_PATIENCE = 4
FINETUNE_PATIENCE = 5

PARAM_GRID = {
    "head_learning_rate": (3e-4, 1e-3),
    "finetune_learning_rate": (1e-5, 3e-5),
    "weight_decay": (1e-4, 1e-3),
    "dropout": (0.2, 0.5),
}

DEFAULT_SEARCH_SEED = 42
DEFAULT_CONFIRMATION_SEEDS = (17, 42, 2026)


@dataclass(frozen=True)
class SearchConfig:
    head_learning_rate: float
    finetune_learning_rate: float
    weight_decay: float
    dropout: float


@dataclass
class PhaseResult:
    state_dict: dict[str, torch.Tensor]
    best_metrics: dict[str, Any]
    best_epoch: int
    history: list[dict[str, Any]]


class RandomAffinePreserveBackground:
    """Apply a small affine transform using the image corners as fill color."""

    def __init__(
        self,
        *,
        degrees: float = 6.0,
        translate: float = 0.04,
        scale: tuple[float, float] = (0.94, 1.06),
        probability: float = 0.75,
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image

        image = image.convert("RGB")
        width, height = image.size
        corners = (
            image.getpixel((0, 0)),
            image.getpixel((width - 1, 0)),
            image.getpixel((0, height - 1)),
            image.getpixel((width - 1, height - 1)),
        )
        fill = tuple(
            round(sum(pixel[channel] for pixel in corners) / len(corners))
            for channel in range(3)
        )

        angle = random.uniform(-self.degrees, self.degrees)
        translation = [
            round(random.uniform(-self.translate, self.translate) * width),
            round(random.uniform(-self.translate, self.translate) * height),
        ]
        scale = random.uniform(*self.scale)

        return TF.affine(
            image,
            angle=angle,
            translate=translation,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=fill,
        )


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def training_transform() -> transforms.Compose:
    """Create conservative augmentation suitable for diagrams and equations."""
    return transforms.Compose(
        [
            transforms.Resize(
                (INPUT_SIZE, INPUT_SIZE),
                interpolation=InterpolationMode.BILINEAR,
            ),
            RandomAffinePreserveBackground(),
            transforms.ColorJitter(
                brightness=0.12,
                contrast=0.15,
                saturation=0.05,
                hue=0.0,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.7))],
                p=0.12,
            ),
            transforms.RandomGrayscale(p=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluation_transform() -> transforms.Compose:
    """Create deterministic validation, testing, and inference transforms."""
    return transforms.Compose(
        [
            transforms.Resize(
                (INPUT_SIZE, INPUT_SIZE),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_data_loaders(
    data_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    dict[str, int],
]:
    """Load all three splits and verify that their class mappings match."""
    train_dataset = datasets.ImageFolder(
        data_dir / "training",
        transform=training_transform(),
    )
    validation_dataset = datasets.ImageFolder(
        data_dir / "validation",
        transform=evaluation_transform(),
    )
    test_dataset = datasets.ImageFolder(
        data_dir / "testing",
        transform=evaluation_transform(),
    )

    class_to_idx = train_dataset.class_to_idx
    if set(class_to_idx) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected classes {sorted(EXPECTED_CLASSES)}, "
            f"found {sorted(class_to_idx)}"
        )
    if validation_dataset.class_to_idx != class_to_idx:
        raise RuntimeError("Validation class mapping differs from training")
    if test_dataset.class_to_idx != class_to_idx:
        raise RuntimeError("Testing class mapping differs from training")

    generator = torch.Generator()
    generator.manual_seed(seed)
    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_loader_args,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_args,
    )
    return train_loader, validation_loader, test_loader, class_to_idx


def build_model(
    *,
    dropout: float,
    number_of_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Create ResNet18 and replace its ImageNet classification layer."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    input_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(input_features, number_of_classes),
    )
    return model


def freeze_for_head_training(model: nn.Module) -> None:
    """Freeze the full backbone and leave only the new head trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_last_block(model: nn.Module) -> None:
    """Fine-tune layer4 and the classification head only."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def keep_batch_norm_fixed(model: nn.Module) -> None:
    """Keep pretrained BatchNorm statistics fixed for the small dataset."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy a model state to CPU so checkpoints do not retain GPU memory."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def classification_metrics(
    targets: list[int],
    predictions: list[int],
    *,
    number_of_classes: int,
) -> dict[str, Any]:
    """Calculate accuracy, macro metrics, and a confusion matrix."""
    confusion = [
        [0 for _ in range(number_of_classes)]
        for _ in range(number_of_classes)
    ]
    for target, prediction in zip(targets, predictions):
        confusion[target][prediction] += 1

    total = len(targets)
    correct = sum(confusion[index][index] for index in range(number_of_classes))
    precision_values = []
    recall_values = []
    f1_values = []

    for class_index in range(number_of_classes):
        true_positive = confusion[class_index][class_index]
        false_positive = sum(
            confusion[row][class_index]
            for row in range(number_of_classes)
            if row != class_index
        )
        false_negative = sum(
            confusion[class_index][column]
            for column in range(number_of_classes)
            if column != class_index
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": mean(precision_values),
        "macro_recall": mean(recall_values),
        "macro_f1": mean(f1_values),
        "per_class_precision": precision_values,
        "per_class_recall": recall_values,
        "per_class_f1": f1_values,
        "confusion_matrix": confusion,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Run one training or evaluation epoch."""
    is_training = optimizer is not None
    if is_training:
        model.train()
        keep_batch_norm_fixed(model)
    else:
        model.eval()

    loss_sum = 0.0
    sample_count = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        targets_all.extend(targets.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())

    metrics = classification_metrics(
        targets_all,
        predictions_all,
        number_of_classes=len(loader.dataset.classes),
    )
    metrics["loss"] = loss_sum / sample_count
    return metrics


def metric_is_better(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
) -> bool:
    """Prefer macro F1, then accuracy, then lower validation loss."""
    if best is None:
        return True
    candidate_key = (
        candidate["macro_f1"],
        candidate["accuracy"],
        -candidate["loss"],
    )
    best_key = (
        best["macro_f1"],
        best["accuracy"],
        -best["loss"],
    )
    return candidate_key > best_key


def fit_phase(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    phase_name: str,
    maximum_epochs: int,
    patience: int,
    use_early_stopping: bool = True,
) -> PhaseResult:
    """Train one phase and restore the epoch with best validation macro F1."""
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, maximum_epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        record = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(record)
        print(
            f"    {phase_name} epoch {epoch:02d}/{maximum_epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={validation_metrics['loss']:.4f}, "
            f"val_acc={validation_metrics['accuracy']:.4f}, "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}"
        )

        if metric_is_better(validation_metrics, best_metrics):
            best_state = clone_state_dict(model)
            best_metrics = copy.deepcopy(validation_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                use_early_stopping
                and epochs_without_improvement >= patience
            ):
                print(f"    Early stopping {phase_name} at epoch {epoch}")
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError(f"No checkpoint produced during {phase_name}")
    model.load_state_dict(best_state)
    return PhaseResult(
        state_dict=best_state,
        best_metrics=best_metrics,
        best_epoch=best_epoch,
        history=history,
    )


def train_trial(
    config: SearchConfig,
    *,
    seed: int,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    return_state: bool = False,
    head_max_epochs: int = HEAD_MAX_EPOCHS,
    finetune_max_epochs: int = FINETUNE_MAX_EPOCHS,
    use_early_stopping: bool = True,
) -> tuple[
    dict[str, Any],
    dict[str, torch.Tensor] | None,
    list[dict[str, Any]],
    dict[str, int],
]:
    """Train one complete frozen-head plus fine-tuning trial."""
    if head_max_epochs <= 0:
        raise ValueError("head_max_epochs must be greater than 0")
    if finetune_max_epochs <= 0:
        raise ValueError("finetune_max_epochs must be greater than 0")

    set_seed(seed)
    train_loader, validation_loader, _, class_to_idx = create_data_loaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    model = build_model(
        dropout=config.dropout,
        number_of_classes=len(class_to_idx),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    start_time = time.perf_counter()

    freeze_for_head_training(model)
    head_optimizer = AdamW(
        model.fc.parameters(),
        lr=config.head_learning_rate,
        weight_decay=config.weight_decay,
    )
    head_result = fit_phase(
        model,
        train_loader,
        validation_loader,
        head_optimizer,
        criterion,
        device,
        phase_name="head",
        maximum_epochs=head_max_epochs,
        patience=HEAD_PATIENCE,
        use_early_stopping=use_early_stopping,
    )

    model.load_state_dict(head_result.state_dict)
    unfreeze_last_block(model)
    finetune_optimizer = AdamW(
        [
            {
                "params": model.layer4.parameters(),
                "lr": config.finetune_learning_rate,
            },
            {
                "params": model.fc.parameters(),
                "lr": config.finetune_learning_rate * 10.0,
            },
        ],
        weight_decay=config.weight_decay,
    )
    finetune_result = fit_phase(
        model,
        train_loader,
        validation_loader,
        finetune_optimizer,
        criterion,
        device,
        phase_name="finetune",
        maximum_epochs=finetune_max_epochs,
        patience=FINETUNE_PATIENCE,
        use_early_stopping=use_early_stopping,
    )

    if metric_is_better(
        finetune_result.best_metrics,
        head_result.best_metrics,
    ):
        selected_result = finetune_result
        selected_phase = "finetune"
    else:
        selected_result = head_result
        selected_phase = "head"

    elapsed_seconds = time.perf_counter() - start_time
    trial_summary = {
        "seed": seed,
        **asdict(config),
        "selected_phase": selected_phase,
        "selected_epoch": selected_result.best_epoch,
        "validation_loss": selected_result.best_metrics["loss"],
        "validation_accuracy": selected_result.best_metrics["accuracy"],
        "validation_macro_precision": selected_result.best_metrics[
            "macro_precision"
        ],
        "validation_macro_recall": selected_result.best_metrics[
            "macro_recall"
        ],
        "validation_macro_f1": selected_result.best_metrics["macro_f1"],
        "elapsed_seconds": elapsed_seconds,
    }
    state = selected_result.state_dict if return_state else None
    history = head_result.history + finetune_result.history

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return trial_summary, state, history, class_to_idx


def parameter_combinations() -> list[SearchConfig]:
    """Expand PARAM_GRID into all 16 SearchConfig objects."""
    keys = tuple(PARAM_GRID)
    return [
        SearchConfig(**dict(zip(keys, values)))
        for values in itertools.product(*(PARAM_GRID[key] for key in keys))
    ]


def trial_sort_key(result: dict[str, Any]) -> tuple[float, float, float]:
    return (
        result["validation_macro_f1"],
        result["validation_accuracy"],
        -result["validation_loss"],
    )


def config_identifier(config: SearchConfig) -> str:
    raw = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:10]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Rewrite the result CSV after every completed trial."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def plot_training_history(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save loss and macro-F1 curves from the final training run."""
    x_values = list(range(1, len(history) + 1))
    head_length = sum(row["phase"] == "head" for row in history)
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(
        x_values,
        [row["train_loss"] for row in history],
        label="training",
    )
    axes[0].plot(
        x_values,
        [row["validation_loss"] for row in history],
        label="validation",
    )
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        x_values,
        [row["train_macro_f1"] for row in history],
        label="training",
    )
    axes[1].plot(
        x_values,
        [row["validation_macro_f1"] for row in history],
        label="validation",
    )
    axes[1].set_xlabel("Epoch across both phases")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    if 0 < head_length < len(history):
        for axis in axes:
            axis.axvline(
                head_length + 0.5,
                color="gray",
                linestyle="--",
                label="fine-tuning starts",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_confusion_matrix(
    confusion: list[list[int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save the final testing confusion matrix."""
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Testing confusion matrix",
    )
    threshold = max(max(row) for row in confusion) / 2.0
    for row_index, row in enumerate(confusion):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and (
        not hasattr(torch.backends, "mps")
        or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search ResNet18 hyperparameters without training the final model."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Use 0 by default for reliable execution on Windows.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=DEFAULT_SEARCH_SEED,
    )
    parser.add_argument(
        "--confirmation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONFIRMATION_SEEDS),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    model_dir = args.model_dir.resolve()
    results_dir = args.results_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    configurations = parameter_combinations()
    if not 1 <= args.top_k <= len(configurations):
        raise ValueError(
            f"top-k must be between 1 and {len(configurations)}"
        )

    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Coarse grid combinations: {len(configurations)}")
    print("Testing data will not be evaluated during parameter search.\n")

    all_rows: list[dict[str, Any]] = []
    coarse_records: list[tuple[SearchConfig, dict[str, Any]]] = []
    csv_path = results_dir / "grid_search_results.csv"

    for trial_number, config in enumerate(configurations, start=1):
        print(
            f"\nCoarse trial {trial_number}/{len(configurations)} "
            f"[{config_identifier(config)}]: {asdict(config)}"
        )
        result, _, _, _ = train_trial(
            config,
            seed=args.search_seed,
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        result["search_stage"] = "coarse"
        result["config_id"] = config_identifier(config)
        all_rows.append(result)
        coarse_records.append((config, result))
        write_csv(all_rows, csv_path)

    coarse_records.sort(
        key=lambda item: trial_sort_key(item[1]),
        reverse=True,
    )
    top_configs = [item[0] for item in coarse_records[: args.top_k]]

    print("\nTop configurations selected for multi-seed confirmation:")
    for rank, config in enumerate(top_configs, start=1):
        print(f"  {rank}. [{config_identifier(config)}] {asdict(config)}")

    confirmation_records: dict[
        SearchConfig,
        list[dict[str, Any]],
    ] = {config: [] for config in top_configs}

    total_confirmation_trials = (
        len(top_configs) * len(args.confirmation_seeds)
    )
    completed = 0
    for config in top_configs:
        for seed in args.confirmation_seeds:
            completed += 1
            print(
                f"\nConfirmation trial {completed}/"
                f"{total_confirmation_trials} "
                f"[{config_identifier(config)}], seed={seed}"
            )
            result, _, _, _ = train_trial(
                config,
                seed=seed,
                data_dir=data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            result["search_stage"] = "confirmation"
            result["config_id"] = config_identifier(config)
            all_rows.append(result)
            confirmation_records[config].append(result)
            write_csv(all_rows, csv_path)

    confirmation_summary = []
    for config, records in confirmation_records.items():
        f1_values = [row["validation_macro_f1"] for row in records]
        accuracy_values = [row["validation_accuracy"] for row in records]
        loss_values = [row["validation_loss"] for row in records]
        confirmation_summary.append(
            {
                "config": config,
                "mean_macro_f1": mean(f1_values),
                "std_macro_f1": pstdev(f1_values),
                "mean_accuracy": mean(accuracy_values),
                "mean_loss": mean(loss_values),
                "runs": records,
            }
        )

    confirmation_summary.sort(
        key=lambda item: (
            item["mean_macro_f1"],
            item["mean_accuracy"],
            -item["mean_loss"],
        ),
        reverse=True,
    )
    winner = confirmation_summary[0]
    best_config: SearchConfig = winner["config"]
    best_confirmation_run = max(
        winner["runs"],
        key=trial_sort_key,
    )
    best_seed = best_confirmation_run["seed"]

    print("\nSelected configuration:")
    print(json.dumps(asdict(best_config), indent=2))
    print(
        f"Confirmation macro F1: "
        f"{winner['mean_macro_f1']:.4f} "
        f"+/- {winner['std_macro_f1']:.4f}"
    )
    print(f"Suggested seed: {best_seed}")

    confirmation_json = [
        {
            "config": asdict(item["config"]),
            "mean_macro_f1": item["mean_macro_f1"],
            "std_macro_f1": item["std_macro_f1"],
            "mean_accuracy": item["mean_accuracy"],
            "mean_loss": item["mean_loss"],
            "runs": item["runs"],
        }
        for item in confirmation_summary
    ]
    save_json(
        {
            "selected_config": asdict(best_config),
            "selected_seed": best_seed,
            "parameter_grid": PARAM_GRID,
            "confirmation_summary": confirmation_json,
        },
        model_dir / "best_hyperparameters.json",
    )

    print("\nHyperparameter search complete.")
    print(f"Trial results: {csv_path}")
    print(
        "Selected parameters: "
        f"{model_dir / 'best_hyperparameters.json'}"
    )
    print("No final model was trained and the testing set was not evaluated.")
    print("Edit the settings in train_best_model.py to train best_model.pth.")


if __name__ == "__main__":
    main()
