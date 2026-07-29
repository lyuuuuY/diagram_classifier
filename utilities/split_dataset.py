"""Split all image sources into training, validation, and testing folders.

Each source directory is shuffled and split independently using an 8:1:1
ratio. Images are copied, so the original datasets remain unchanged.

Default usage:

    .\.venv\Scripts\python.exe utilities\split_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "merged_data"
DEFAULT_SEED = 20260729
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("training", "validation", "testing")
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ImageSource:
    label: str
    name: str
    directory: Path


SOURCES = (
    ImageSource(
        label="lewis",
        name="converted",
        directory=PROJECT_ROOT / "data_lewis" / "converted",
    ),
    ImageSource(
        label="lewis",
        name="generated",
        directory=PROJECT_ROOT / "data_lewis" / "generated",
    ),
    ImageSource(
        label="lewis",
        name="handwritten",
        directory=PROJECT_ROOT / "data_lewis" / "han_written_split",
    ),
    ImageSource(
        label="graph",
        name="generated",
        directory=PROJECT_ROOT / "data_Connected_graph" / "generated",
    ),
    ImageSource(
        label="graph",
        name="generated-style2",
        directory=PROJECT_ROOT
        / "data_Connected_graph"
        / "generated_style2",
    ),
    ImageSource(
        label="graph",
        name="handwritten",
        directory=PROJECT_ROOT
        / "data_Connected_graph"
        / "han_written_split",
    ),
    ImageSource(
        label="equation",
        name="generated",
        directory=PROJECT_ROOT
        / "data_calculus_equations"
        / "generated",
    ),
    ImageSource(
        label="equation",
        name="handwritten",
        directory=PROJECT_ROOT
        / "data_calculus_equations"
        / "han_written_split",
    ),
)


def _image_files(directory: Path) -> list[Path]:
    """Return supported image files recursively in deterministic order."""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _split_counts(total: int) -> tuple[int, int, int]:
    """Allocate integer counts by the largest-remainder method."""
    raw_counts = [total * ratio for ratio in SPLIT_RATIOS]
    counts = [int(value) for value in raw_counts]
    remaining = total - sum(counts)
    priority = sorted(
        range(len(counts)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in priority[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _source_seed(seed: int, source: ImageSource) -> int:
    """Derive a stable independent random seed for one source."""
    identity = f"{seed}:{source.label}:{source.name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")


def _prepare_output(output_dir: Path) -> None:
    """Clear only the three managed split folders, then recreate class dirs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for label in ("lewis", "graph", "equation"):
            (split_dir / label).mkdir(parents=True, exist_ok=True)


def split_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, int]]:
    """Copy all configured sources into stratified dataset folders."""
    missing = [source.directory for source in SOURCES if not source.directory.is_dir()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing source directories:\n{missing_text}")

    source_files: dict[ImageSource, list[Path]] = {}
    for source in SOURCES:
        files = _image_files(source.directory)
        if not files:
            raise RuntimeError(f"No images found in {source.directory}")
        source_files[source] = files

    destination_root = Path(output_dir).resolve()
    _prepare_output(destination_root)
    summary = {
        split_name: {"lewis": 0, "graph": 0, "equation": 0}
        for split_name in SPLIT_NAMES
    }

    for source in SOURCES:
        files = source_files[source].copy()
        random.Random(_source_seed(seed, source)).shuffle(files)
        train_count, validation_count, test_count = _split_counts(len(files))
        boundaries = (
            train_count,
            train_count + validation_count,
            train_count + validation_count + test_count,
        )
        split_files = (
            files[: boundaries[0]],
            files[boundaries[0] : boundaries[1]],
            files[boundaries[1] : boundaries[2]],
        )

        counts_text = ", ".join(
            f"{split_name}={len(items)}"
            for split_name, items in zip(SPLIT_NAMES, split_files)
        )
        print(
            f"{source.label}/{source.name}: {len(files)} images "
            f"({counts_text})"
        )

        for split_name, items in zip(SPLIT_NAMES, split_files):
            class_dir = destination_root / split_name / source.label
            for source_index, source_path in enumerate(items, start=1):
                destination_name = (
                    f"{source.label}-{source.name}-{source_index:04d}-"
                    f"{source_path.name}"
                )
                shutil.copy2(source_path, class_dir / destination_name)
            summary[split_name][source.label] += len(items)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split image sources into an 8:1:1 merged dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"shuffle seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = split_dataset(args.output_dir, seed=args.seed)
    print("\nFinal counts:")
    for split_name in SPLIT_NAMES:
        class_counts = summary[split_name]
        total = sum(class_counts.values())
        print(
            f"  {split_name}: {class_counts} "
            f"(total={total})"
        )


if __name__ == "__main__":
    main()
