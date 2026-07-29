r"""Split handwritten Lewis-structure grid pages into classifier PNG files.

Default usage:

    .\.venv\Scripts\python.exe utilities\split_handwritten.py

Input:
    data_lewis\hand_written

Output:
    data_lewis\han_written_split_training
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data_calculus_equations" / "hand_written"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_calculus_equations" / "han_written_split"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
OUTPUT_SIZE = 224


def _group_adjacent(indices: Iterable[int], max_gap: int = 2) -> list[tuple[int, int]]:
    groups: list[list[int]] = []
    for index in indices:
        if not groups or index - groups[-1][-1] > max_gap:
            groups.append([index])
        else:
            groups[-1].append(index)
    return [(group[0], group[-1]) for group in groups]


def _detect_grid_lines(
    grayscale: Image.Image,
    *,
    axis: str,
    minimum_fraction: float = 0.55,
) -> list[int]:
    """Return centers of long dark horizontal or vertical grid lines."""
    width, height = grayscale.size
    pixels = grayscale.load()

    if axis == "horizontal":
        scores = [
            sum(pixels[x, y] < 150 for x in range(width))
            for y in range(height)
        ]
        required = width * minimum_fraction
    elif axis == "vertical":
        scores = [
            sum(pixels[x, y] < 150 for y in range(height))
            for x in range(width)
        ]
        required = height * minimum_fraction
    else:
        raise ValueError("axis must be 'horizontal' or 'vertical'")

    runs = _group_adjacent(
        (index for index, score in enumerate(scores) if score >= required)
    )
    return [round((start + end) / 2) for start, end in runs]


def _grid_boundaries(
    length: int,
    detected_lines: list[int],
    *,
    cell_count: int,
) -> list[int]:
    """Use detected internal lines, falling back to an equal grid."""
    internal_lines = [line for line in detected_lines if 2 < line < length - 3]
    expected_internal_count = cell_count - 1

    if len(internal_lines) == expected_internal_count:
        return [0, *internal_lines, length]

    return [round(index * length / cell_count) for index in range(cell_count + 1)]


def _binary_ink_mask(image: Image.Image, threshold: int = 205) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    normalized = ImageOps.autocontrast(grayscale, cutoff=1)
    return normalized.point(lambda value: 255 if value < threshold else 0)


def _connected_components(
    mask: Image.Image,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Return (pixel count, bounding box) for 8-connected ink components."""
    width, height = mask.size
    pixels = mask.load()
    visited = bytearray(width * height)
    components: list[tuple[int, tuple[int, int, int, int]]] = []

    for y in range(height):
        for x in range(width):
            flat_index = y * width + x
            if visited[flat_index] or pixels[x, y] == 0:
                continue

            queue = deque([(x, y)])
            visited[flat_index] = 1
            area = 0
            left = right = x
            top = bottom = y

            while queue:
                current_x, current_y = queue.popleft()
                area += 1
                left = min(left, current_x)
                right = max(right, current_x)
                top = min(top, current_y)
                bottom = max(bottom, current_y)

                for neighbor_y in range(
                    max(0, current_y - 1),
                    min(height, current_y + 2),
                ):
                    for neighbor_x in range(
                        max(0, current_x - 1),
                        min(width, current_x + 2),
                    ):
                        neighbor_index = neighbor_y * width + neighbor_x
                        if (
                            not visited[neighbor_index]
                            and pixels[neighbor_x, neighbor_y] != 0
                        ):
                            visited[neighbor_index] = 1
                            queue.append((neighbor_x, neighbor_y))

            components.append((area, (left, top, right + 1, bottom + 1)))

    return components


def _union_boxes(
    boxes: Iterable[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    boxes = list(boxes)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _content_box(cell: Image.Image) -> tuple[int, int, int, int] | None:
    """Find handwriting while ignoring isolated scan marks and grid remnants."""
    width, height = cell.size
    edge_x = max(4, round(width * 0.025))
    edge_y = max(4, round(height * 0.04))
    inner_box = (edge_x, edge_y, width - edge_x, height - edge_y)
    inner = cell.crop(inner_box)
    components = _connected_components(_binary_ink_mask(inner))

    # Letters and bonds form the larger components. Their union defines the
    # main diagram; nearby small components are retained as lone-pair dots.
    significant = [
        box
        for area, box in components
        if area >= 12 or (box[2] - box[0]) >= 10 or (box[3] - box[1]) >= 10
    ]
    main_box = _union_boxes(significant)
    if main_box is None:
        return None

    main_width = main_box[2] - main_box[0]
    main_height = main_box[3] - main_box[1]
    nearby_distance = max(24, round(max(main_width, main_height) * 0.18))
    nearby_box = (
        main_box[0] - nearby_distance,
        main_box[1] - nearby_distance,
        main_box[2] + nearby_distance,
        main_box[3] + nearby_distance,
    )

    retained_boxes = []
    retained_ink_pixels = 0
    for area, box in components:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        if (
            nearby_box[0] <= center_x <= nearby_box[2]
            and nearby_box[1] <= center_y <= nearby_box[3]
        ):
            retained_boxes.append(box)
            retained_ink_pixels += area

    if retained_ink_pixels < 35:
        return None

    content = _union_boxes(retained_boxes)
    if content is None:
        return None

    left, top, right, bottom = content
    content_width = right - left
    content_height = bottom - top
    padding_x = max(8, round(content_width * 0.12))
    padding_y = max(8, round(content_height * 0.12))

    return (
        max(0, left + inner_box[0] - padding_x),
        max(0, top + inner_box[1] - padding_y),
        min(width, right + inner_box[0] + padding_x),
        min(height, bottom + inner_box[1] + padding_y),
    )


def _background_color(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    sample_size = max(1, min(width, height) // 10)
    corner_samples = (
        image.crop((0, 0, sample_size, sample_size)),
        image.crop((width - sample_size, 0, width, sample_size)),
        image.crop((0, height - sample_size, sample_size, height)),
        image.crop(
            (
                width - sample_size,
                height - sample_size,
                width,
                height,
            )
        ),
    )
    pixels: list[tuple[int, int, int]] = []
    for sample in corner_samples:
        colors = sample.getcolors(maxcolors=sample.width * sample.height)
        if colors:
            pixels.append(max(colors, key=lambda item: item[0])[1])

    if not pixels:
        return (255, 255, 255)
    return tuple(
        round(sum(color[channel] for color in pixels) / len(pixels))
        for channel in range(3)
    )


def _fit_to_classifier_format(
    image: Image.Image,
    *,
    size: int = OUTPUT_SIZE,
    margin: int = 12,
) -> Image.Image:
    image = image.convert("RGB")
    background = _background_color(image)
    fitted = image.copy()
    fitted.thumbnail((size - 2 * margin, size - 2 * margin), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (size, size), background)
    left = (size - fitted.width) // 2
    top = (size - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def split_page(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    rows: int = 6,
    columns: int = 2,
    size: int = OUTPUT_SIZE,
) -> list[Path]:
    """Split one grid page, skipping cells without handwriting."""
    source_path = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as source:
        page = source.convert("RGB")

    grayscale = ImageOps.grayscale(page)
    horizontal_lines = _detect_grid_lines(grayscale, axis="horizontal")
    vertical_lines = _detect_grid_lines(grayscale, axis="vertical")
    x_boundaries = _grid_boundaries(
        page.width,
        vertical_lines,
        cell_count=columns,
    )
    y_boundaries = _grid_boundaries(
        page.height,
        horizontal_lines,
        cell_count=rows,
    )

    output_paths: list[Path] = []
    line_margin = 3

    for row in range(rows):
        for column in range(columns):
            left = x_boundaries[column] + (line_margin if column > 0 else 0)
            right = x_boundaries[column + 1] - (
                line_margin if column < columns - 1 else 0
            )
            top = y_boundaries[row] + (line_margin if row > 0 else 0)
            bottom = y_boundaries[row + 1] - (
                line_margin if row < rows - 1 else 0
            )
            cell = page.crop((left, top, right, bottom))
            content_box = _content_box(cell)
            if content_box is None:
                print(
                    f"Skipped empty cell: {source_path.name} "
                    f"row={row + 1} column={column + 1}"
                )
                continue

            content = cell.crop(content_box)
            converted = _fit_to_classifier_format(content, size=size)
            output_name = (
                f"handwritten-{source_path.stem.lower()}-"
                f"r{row + 1:02d}-c{column + 1:02d}.png"
            )
            output_path = destination / output_name
            converted.save(output_path, format="PNG", optimize=True)
            output_paths.append(output_path)
            print(f"Saved {output_path.name}")

    return output_paths


def split_directory(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    rows: int = 6,
    columns: int = 2,
    size: int = OUTPUT_SIZE,
) -> list[Path]:
    source_dir = Path(input_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {source_dir}")

    source_paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not source_paths:
        raise FileNotFoundError(f"No supported images found in: {source_dir}")

    output_paths: list[Path] = []
    for source_path in source_paths:
        print(f"Processing {source_path.name}")
        output_paths.extend(
            split_page(
                source_path,
                output_dir,
                rows=rows,
                columns=columns,
                size=size,
            )
        )
    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split handwritten Lewis grid pages into square PNG files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--size", type=int, default=OUTPUT_SIZE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_paths = split_directory(
        args.input_dir,
        args.output_dir,
        rows=args.rows,
        columns=args.columns,
        size=args.size,
    )
    print(
        f"Created {len(output_paths)} images in "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
