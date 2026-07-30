"""Train and test one final model using manually entered hyperparameters.

Edit the values in the ``MANUAL SETTINGS`` section below before running:

    .\.venv\Scripts\python.exe train_best_model.py

This script trains one model, selects its best validation checkpoint, evaluates
the testing set once, and saves a checkpoint compatible with ``classifier.py``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from parameter_search import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    PROJECT_ROOT,
    SearchConfig,
    build_model,
    create_data_loaders,
    plot_confusion_matrix,
    plot_training_history,
    resolve_device,
    run_epoch,
    save_json,
    train_trial,
)


HEAD_LEARNING_RATE = 0.0003
FINETUNE_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-3
DROPOUT = 0.5
SEED = 17

HEAD_EPOCHS = 10
FINETUNE_EPOCHS = 14

BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = "auto"  

DATA_DIR = PROJECT_ROOT / "merged_data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "training_results"
MODEL_FILENAME = "best_model.pth"


def validate_manual_settings(config: SearchConfig) -> None:
    """Reject invalid hand-entered settings before starting a long run."""
    if config.head_learning_rate <= 0:
        raise ValueError("HEAD_LEARNING_RATE must be greater than 0")
    if config.finetune_learning_rate <= 0:
        raise ValueError("FINETUNE_LEARNING_RATE must be greater than 0")
    if config.weight_decay < 0:
        raise ValueError("WEIGHT_DECAY must be at least 0")
    if not 0 <= config.dropout < 1:
        raise ValueError("DROPOUT must be in the range [0, 1)")
    if HEAD_EPOCHS <= 0:
        raise ValueError("HEAD_EPOCHS must be greater than 0")
    if FINETUNE_EPOCHS <= 0:
        raise ValueError("FINETUNE_EPOCHS must be greater than 0")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be greater than 0")
    if NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS must be at least 0")
    if not MODEL_FILENAME.lower().endswith(".pth"):
        raise ValueError("MODEL_FILENAME must end with .pth")


def ordered_class_names(class_to_idx: dict[str, int]) -> list[str]:
    """Return class names in the exact output-index order used by the model."""
    return [
        class_name
        for class_name, _ in sorted(
            class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]


def checkpoint_data(
    *,
    state_dict: dict[str, torch.Tensor],
    class_to_idx: dict[str, int],
    config: SearchConfig,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the checkpoint format consumed by classifier.py and evaluate.py."""
    return {
        "architecture": "resnet18",
        "model_state_dict": state_dict,
        "class_to_idx": class_to_idx,
        "class_names": ordered_class_names(class_to_idx),
        "input_size": INPUT_SIZE,
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "dropout": config.dropout,
        "hyperparameters": asdict(config),
        "seed": SEED,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }


def main() -> None:
    config = SearchConfig(
        head_learning_rate=HEAD_LEARNING_RATE,
        finetune_learning_rate=FINETUNE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
    )
    validate_manual_settings(config)

    data_dir = Path(DATA_DIR).resolve()
    model_dir = Path(MODEL_DIR).resolve()
    results_dir = Path(RESULTS_DIR).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(DEVICE)

    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Manual hyperparameters: {asdict(config)}")
    print(f"Seed: {SEED}")
    print(f"Head epochs (fixed): {HEAD_EPOCHS}")
    print(f"Fine-tuning epochs (fixed): {FINETUNE_EPOCHS}")
    print("\nTraining the final model...")

    validation_metrics, final_state, training_history, class_to_idx = (
        train_trial(
            config,
            seed=SEED,
            data_dir=data_dir,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=device,
            return_state=True,
            head_max_epochs=HEAD_EPOCHS,
            finetune_max_epochs=FINETUNE_EPOCHS,
            use_early_stopping=False,
        )
    )
    if final_state is None:
        raise RuntimeError("Final training did not return model weights")

    _, _, test_loader, test_class_to_idx = create_data_loaders(
        data_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        pin_memory=device.type == "cuda",
    )
    if test_class_to_idx != class_to_idx:
        raise RuntimeError("Class mapping changed before testing")

    final_model = build_model(
        dropout=config.dropout,
        number_of_classes=len(class_to_idx),
        pretrained=False,
    ).to(device)
    final_model.load_state_dict(final_state)

    print("\nEvaluating the testing set once...")
    test_metrics = run_epoch(
        final_model,
        test_loader,
        nn.CrossEntropyLoss(),
        device,
    )

    checkpoint = checkpoint_data(
        state_dict=final_state,
        class_to_idx=class_to_idx,
        config=config,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )
    model_path = model_dir / MODEL_FILENAME
    torch.save(checkpoint, model_path)

    class_names = ordered_class_names(class_to_idx)
    save_json(
        {
            "class_names": class_names,
            "class_to_idx": class_to_idx,
        },
        model_dir / "class_names.json",
    )
    save_json(
        {
            "hyperparameters": asdict(config),
            "seed": SEED,
            "head_epochs": HEAD_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "early_stopping": False,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "device": str(device),
            "data_dir": str(data_dir),
            "model_path": str(model_path),
        },
        results_dir / "final_training_config.json",
    )
    save_json(test_metrics, results_dir / "test_metrics.json")
    save_json(
        training_history,
        results_dir / "final_training_history.json",
    )
    plot_training_history(
        training_history,
        results_dir / "training_curves.png",
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        results_dir / "confusion_matrix.png",
    )

    print("\nFinal training complete.")
    print(f"Best model: {model_path}")
    print(
        "Validation macro F1: "
        f"{validation_metrics['validation_macro_f1']:.4f}"
    )
    print(f"Testing accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Testing macro F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
