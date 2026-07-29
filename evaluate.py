"""Evaluate and visualize the trained best model on the testing set.

This script is intentionally separate from ``train_best_model.py``. It loads
``models/best_model.pth``, evaluates ``merged_data/testing`` once, prints the
metrics, and saves visual reports.

Run after training:

    .\.venv\Scripts\python.exe evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DEFAULT_TEST_DIR = PROJECT_ROOT / "merged_data" / "testing"
DEFAULT_NEW_SCOPE_DIR = (
    PROJECT_ROOT / "merged_data" / "testing_new_scope"
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "test_results"


def resolve_device(requested: str) -> torch.device:
    """Select CUDA automatically when it is available."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_checkpoint(
    model_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load and validate the checkpoint produced by train_best_model.py."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Best model not found: {model_path}\n"
            "Run train_best_model.py before evaluate.py."
        )
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    required_keys = {
        "architecture",
        "model_state_dict",
        "class_to_idx",
        "class_names",
        "input_size",
        "normalization",
        "dropout",
    }
    missing = required_keys - set(checkpoint)
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing required fields: {sorted(missing)}"
        )
    if checkpoint["architecture"] != "resnet18":
        raise RuntimeError(
            f"Unsupported architecture: {checkpoint['architecture']}"
        )
    return checkpoint


def build_model(checkpoint: dict[str, Any], device: torch.device) -> nn.Module:
    """Reconstruct exactly the model architecture used during training."""
    class_names = checkpoint["class_names"]
    model = resnet18(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=float(checkpoint["dropout"])),
        nn.Linear(input_features, len(class_names)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


class FlatPrefixDataset(Dataset):
    """Read flat PNG files whose filename prefix provides the true label."""

    def __init__(
        self,
        root: Path,
        *,
        transform: transforms.Compose,
        class_to_idx: dict[str, int],
    ) -> None:
        self.root = root
        self.transform = transform
        self.class_to_idx = dict(class_to_idx)
        self.classes = [
            class_name
            for class_name, _ in sorted(
                self.class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]
        self.samples: list[tuple[str, int]] = []

        image_paths = sorted(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        if not image_paths:
            raise RuntimeError(f"No PNG files found in {root}")

        for image_path in image_paths:
            filename = image_path.stem.lower()
            matching_labels = [
                label
                for label in self.class_to_idx
                if filename.startswith(label)
            ]
            if len(matching_labels) != 1:
                raise RuntimeError(
                    "A flat testing filename must start with exactly one "
                    f"class name {sorted(self.class_to_idx)}: "
                    f"{image_path.name}"
                )
            label = matching_labels[0]
            self.samples.append(
                (str(image_path), self.class_to_idx[label])
            )

        present_labels = {
            label_index for _, label_index in self.samples
        }
        expected_labels = set(self.class_to_idx.values())
        if present_labels != expected_labels:
            raise RuntimeError(
                "Flat testing data must contain every class. "
                f"Found label indexes {sorted(present_labels)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, target = self.samples[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        return self.transform(image), target


def create_test_loader(
    test_dir: Path,
    checkpoint: dict[str, Any],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, Dataset]:
    """Create a loader for class folders or flat class-prefixed PNGs."""
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Testing directory not found: {test_dir}")

    input_size = int(checkpoint["input_size"])
    normalization = checkpoint["normalization"]
    transform = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                normalization["mean"],
                normalization["std"],
            ),
        ]
    )

    class_to_idx = checkpoint["class_to_idx"]
    class_directories = {
        class_name: (test_dir / class_name).is_dir()
        for class_name in class_to_idx
    }
    if all(class_directories.values()):
        dataset: Dataset = datasets.ImageFolder(
            test_dir,
            transform=transform,
        )
    elif any(class_directories.values()):
        raise RuntimeError(
            "Testing data has only some class directories: "
            f"{class_directories}"
        )
    else:
        dataset = FlatPrefixDataset(
            test_dir,
            transform=transform,
            class_to_idx=class_to_idx,
        )

    if dataset.class_to_idx != checkpoint["class_to_idx"]:
        raise RuntimeError(
            "Testing class mapping differs from the checkpoint.\n"
            f"Checkpoint: {checkpoint['class_to_idx']}\n"
            f"Testing: {dataset.class_to_idx}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return loader, dataset


def classification_metrics(
    targets: list[int],
    predictions: list[int],
    *,
    number_of_classes: int,
) -> dict[str, Any]:
    """Calculate overall, per-class, and confusion-matrix metrics."""
    confusion = [
        [0 for _ in range(number_of_classes)]
        for _ in range(number_of_classes)
    ]
    for target, prediction in zip(targets, predictions):
        confusion[target][prediction] += 1

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

    total = len(targets)
    correct = sum(confusion[index][index] for index in range(number_of_classes))
    return {
        "sample_count": total,
        "correct_count": correct,
        "error_count": total - correct,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": mean(precision_values),
        "macro_recall": mean(recall_values),
        "macro_f1": mean(f1_values),
        "per_class_precision": precision_values,
        "per_class_recall": recall_values,
        "per_class_f1": f1_values,
        "confusion_matrix": confusion,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    dataset: Dataset,
    class_names: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the testing set once and retain each image prediction."""
    criterion = nn.CrossEntropyLoss()
    targets_all: list[int] = []
    predictions_all: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    sample_count = 0

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            batch_size = targets.size(0)
            batch_start = sample_count
            loss_sum += loss.item() * batch_size
            sample_count += batch_size

            targets_cpu = targets.cpu().tolist()
            predictions_cpu = predictions.cpu().tolist()
            confidences_cpu = confidences.cpu().tolist()
            probabilities_cpu = probabilities.cpu().tolist()
            targets_all.extend(targets_cpu)
            predictions_all.extend(predictions_cpu)

            for local_index in range(batch_size):
                dataset_index = batch_start + local_index
                source_path = Path(dataset.samples[dataset_index][0]).resolve()
                true_index = targets_cpu[local_index]
                predicted_index = predictions_cpu[local_index]
                row: dict[str, Any] = {
                    "file_path": str(source_path),
                    "true_label": class_names[true_index],
                    "predicted_label": class_names[predicted_index],
                    "confidence": confidences_cpu[local_index],
                    "correct": true_index == predicted_index,
                }
                for class_index, class_name in enumerate(class_names):
                    row[f"probability_{class_name}"] = probabilities_cpu[
                        local_index
                    ][class_index]
                prediction_rows.append(row)

    metrics = classification_metrics(
        targets_all,
        predictions_all,
        number_of_classes=len(class_names),
    )
    metrics["loss"] = loss_sum / sample_count
    return metrics, prediction_rows


def draw_confusion_matrix(
    axis: plt.Axes,
    confusion: list[list[int]],
    class_names: list[str],
) -> None:
    """Draw a labeled confusion matrix on an existing Matplotlib axis."""
    image = axis.imshow(confusion, cmap="Blues")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Confusion matrix",
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


def draw_class_metrics(
    axis: plt.Axes,
    metrics: dict[str, Any],
    class_names: list[str],
) -> None:
    """Draw per-class precision, recall, and F1 bars."""
    x_values = np.arange(len(class_names))
    width = 0.24
    axis.bar(
        x_values - width,
        metrics["per_class_precision"],
        width,
        label="Precision",
    )
    axis.bar(
        x_values,
        metrics["per_class_recall"],
        width,
        label="Recall",
    )
    axis.bar(
        x_values + width,
        metrics["per_class_f1"],
        width,
        label="F1",
    )
    axis.axhline(
        metrics["accuracy"],
        color="black",
        linestyle="--",
        label=f"Accuracy: {metrics['accuracy']:.3f}",
    )
    axis.set(
        xticks=x_values,
        xticklabels=class_names,
        ylim=(0.0, 1.05),
        ylabel="Score",
        title="Metrics by class",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")


def save_summary_plot(
    metrics: dict[str, Any],
    class_names: list[str],
    output_path: Path,
    *,
    dataset_name: str,
) -> None:
    """Save one compact overview of the complete testing result."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    draw_confusion_matrix(
        axes[0],
        metrics["confusion_matrix"],
        class_names,
    )
    draw_class_metrics(axes[1], metrics, class_names)
    figure.suptitle(
        f"{dataset_name} — {metrics['correct_count']}/"
        f"{metrics['sample_count']} correct, "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"macro F1={metrics['macro_f1']:.3f}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def save_prediction_gallery(
    predictions: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    incorrect_only: bool,
    maximum_images: int,
) -> None:
    """Save a grid with image, true class, prediction, and confidence."""
    selected = (
        [row for row in predictions if not row["correct"]]
        if incorrect_only
        else predictions.copy()
    )
    if incorrect_only:
        selected.sort(key=lambda row: row["confidence"], reverse=True)
    selected = selected[:maximum_images]

    if not selected:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No misclassified testing images.",
            ha="center",
            va="center",
            fontsize=16,
            color="#18733c",
        )
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(output_path, dpi=170)
        plt.close(figure)
        return

    column_count = min(6, len(selected))
    row_count = math.ceil(len(selected) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(column_count * 2.45, row_count * 2.65),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")

    for axis, prediction in zip(axes.flat, selected):
        with Image.open(prediction["file_path"]) as source_image:
            image = source_image.convert("RGB")
            axis.imshow(image)
            width, height = image.size
        color = "#18864b" if prediction["correct"] else "#c62828"
        axis.add_patch(
            Rectangle(
                (-0.5, -0.5),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=4,
            )
        )
        axis.set_title(
            f"True: {prediction['true_label']}\n"
            f"Pred: {prediction['predicted_label']} "
            f"({prediction['confidence']:.1%})",
            color=color,
            fontsize=9,
        )

    figure.suptitle(title, fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def save_json(data: Any, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def save_prediction_csv(
    predictions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(predictions[0]),
        )
        writer.writeheader()
        writer.writerows(predictions)


def print_metrics(
    metrics: dict[str, Any],
    class_names: list[str],
    *,
    dataset_name: str,
) -> None:
    """Print a readable testing report in the terminal."""
    print(f"\n{dataset_name} results")
    print("=" * 68)
    print(f"Samples:         {metrics['sample_count']}")
    print(f"Correct:         {metrics['correct_count']}")
    print(f"Errors:          {metrics['error_count']}")
    print(f"Loss:            {metrics['loss']:.6f}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print("\nPer-class metrics")
    print(f"{'Class':<12}{'Precision':>12}{'Recall':>12}{'F1':>12}")
    for index, class_name in enumerate(class_names):
        print(
            f"{class_name:<12}"
            f"{metrics['per_class_precision'][index]:>12.4f}"
            f"{metrics['per_class_recall'][index]:>12.4f}"
            f"{metrics['per_class_f1'][index]:>12.4f}"
        )
    print("\nConfusion matrix (rows=true, columns=predicted)")
    print("Classes:", class_names)
    for row in metrics["confusion_matrix"]:
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize the trained best model."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
    )
    parser.add_argument(
        "--new-scope-dir",
        type=Path,
        default=DEFAULT_NEW_SCOPE_DIR,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--maximum-gallery-images",
        type=int,
        default=60,
    )
    return parser.parse_args()


def evaluate_scope(
    *,
    scope_name: str,
    test_dir: Path,
    results_dir: Path,
    model: nn.Module,
    checkpoint: dict[str, Any],
    class_names: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    maximum_gallery_images: int,
) -> dict[str, Any]:
    """Evaluate one scope and save all of its files in one directory."""
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nEvaluating {scope_name}: {test_dir}")

    test_loader, test_dataset = create_test_loader(
        test_dir,
        checkpoint,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics, predictions = evaluate(
        model,
        test_loader,
        test_dataset,
        class_names,
        device,
    )

    print_metrics(
        metrics,
        class_names,
        dataset_name=scope_name,
    )
    save_json(metrics, results_dir / "test_metrics.json")
    save_prediction_csv(
        predictions,
        results_dir / "test_predictions.csv",
    )
    save_summary_plot(
        metrics,
        class_names,
        results_dir / "test_summary.png",
        dataset_name=scope_name,
    )
    save_prediction_gallery(
        predictions,
        results_dir / "test_predictions_gallery.png",
        title=f"All {scope_name} predictions",
        incorrect_only=False,
        maximum_images=maximum_gallery_images,
    )
    save_prediction_gallery(
        predictions,
        results_dir / "misclassified_examples.png",
        title=f"Misclassified {scope_name} images",
        incorrect_only=True,
        maximum_images=maximum_gallery_images,
    )

    print(f"\nSaved {scope_name} report to: {results_dir}")
    return metrics


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_path = args.model_path.resolve()
    test_dir = args.test_dir.resolve()
    new_scope_dir = args.new_scope_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {model_path}")

    checkpoint = load_checkpoint(model_path, device)
    class_names = list(checkpoint["class_names"])
    model = build_model(checkpoint, device)

    standard_metrics = evaluate_scope(
        scope_name="testing",
        test_dir=test_dir,
        results_dir=results_dir / "testing",
        model=model,
        checkpoint=checkpoint,
        class_names=class_names,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        maximum_gallery_images=args.maximum_gallery_images,
    )
    new_scope_metrics = evaluate_scope(
        scope_name="testing_new_scope",
        test_dir=new_scope_dir,
        results_dir=results_dir / "testing_new_scope",
        model=model,
        checkpoint=checkpoint,
        class_names=class_names,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        maximum_gallery_images=args.maximum_gallery_images,
    )

    print("\nEvaluation complete")
    print(
        f"  testing: accuracy={standard_metrics['accuracy']:.4f}, "
        f"macro_f1={standard_metrics['macro_f1']:.4f}"
    )
    print(
        "  testing_new_scope: "
        f"accuracy={new_scope_metrics['accuracy']:.4f}, "
        f"macro_f1={new_scope_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
