from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops


DEFAULT_INPUT_DIR = "merged_data"
OUTPUT_SIZE = 224


def _dominant_background(image: Image.Image) -> tuple[int, int, int]:
    """Estimate the background colour from a small copy of the whole image."""
    sample = image.copy()
    sample.thumbnail((64, 64), Image.Resampling.BOX)
    colors = sample.getcolors(maxcolors=sample.width * sample.height)
    if colors is None:
        raise RuntimeError("Could not determine the image background colour")
    return max(colors, key=lambda item: item[0])[1]


def _remove_bottom_annotations(
    image: Image.Image,
    *,
    background: tuple[int, int, int],
    start_ratio: float = 0.65,
) -> Image.Image:
    """Remove the blue-grey explanatory text below the main diagram.

    The generated images use a consistent blue-grey colour for captions.
    Matching the colour relative to the background removes anti-aliased text
    while preserving coloured atoms, bonds, brackets, and electron dots that
    may extend into the same vertical area.
    """
    if not 0 <= start_ratio <= 1:
        raise ValueError("start_ratio must be between zero and one")

    cleaned = image.copy()
    pixels = cleaned.load()
    background_red, background_green, background_blue = background
    start_y = round(cleaned.height * start_ratio)

    for y in range(start_y, cleaned.height):
        for x in range(cleaned.width):
            red, green, blue = pixels[x, y]
            delta_red = red - background_red
            delta_green = green - background_green
            delta_blue = blue - background_blue

            if delta_blue < 8:
                continue

            red_ratio = delta_red / delta_blue
            green_ratio = delta_green / delta_blue
            is_annotation_colour = (
                0.68 <= red_ratio <= 0.84
                and 0.81 <= green_ratio <= 0.93
            )
            if is_annotation_colour:
                pixels[x, y] = background

    return cleaned


def _crop_to_content(
    image: Image.Image,
    *,
    background: tuple[int, int, int],
    threshold: int = 12,
    padding_ratio: float = 0.06,
) -> Image.Image:
    """Crop empty background without cutting off wide or tall structures.

    A fixed crop is unsafe because molecules have different aspect ratios.
    This function finds pixels that differ from the dominant background,
    excludes only the thin decorative outer border, and adds padding around
    the detected content.
    """
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative")

    width, height = image.size
    background_image = Image.new("RGB", image.size, background)
    difference = ImageChops.difference(image, background_image)
    red, green, blue = difference.split()
    strongest_difference = ImageChops.lighter(
        ImageChops.lighter(red, green),
        blue,
    )
    foreground_mask = strongest_difference.point(
        lambda value: 255 if value >= threshold else 0
    )

    # The generated cards have a thin dashed border at the outer edge. Search
    # inside it, while retaining enough vertical space for large structures.
    border_x = max(1, round(width * 0.03))
    border_y = max(1, round(height * 0.03))
    search_box = (
        border_x,
        border_y,
        width - border_x,
        height - border_y,
    )
    local_box = foreground_mask.crop(search_box).getbbox()
    if local_box is None:
        return image

    left = local_box[0] + search_box[0]
    top = local_box[1] + search_box[1]
    right = local_box[2] + search_box[0]
    bottom = local_box[3] + search_box[1]

    content_width = right - left
    content_height = bottom - top
    padding_x = max(4, round(content_width * padding_ratio))
    padding_y = max(4, round(content_height * padding_ratio))

    crop_box = (
        max(search_box[0], left - padding_x),
        max(search_box[1], top - padding_y),
        min(search_box[2], right + padding_x),
        min(search_box[3], bottom + padding_y),
    )
    return image.crop(crop_box)


def _fit_on_square(
    image: Image.Image,
    *,
    size: int,
    background: tuple[int, int, int],
    margin: int,
) -> Image.Image:
    """Resize without distortion and center the result on a square canvas."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if margin < 0 or margin * 2 >= size:
        raise ValueError("margin must be non-negative and less than half the size")

    fitted = image.copy()
    fitted.thumbnail(
        (size - 2 * margin, size - 2 * margin),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("RGB", (size, size), background)
    left = (size - fitted.width) // 2
    top = (size - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def convert_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    size: int = OUTPUT_SIZE,
    margin: int = 10,
    auto_crop: bool = True,
    remove_annotations: bool = True,
) -> Path:
    """Convert one image to an RGB square PNG suitable for the classifier."""
    source_path = Path(input_path)
    destination_path = Path(output_path)

    if not source_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {source_path}")

    with Image.open(source_path) as source:
        image = source.convert("RGB")

    background = _dominant_background(image)
    if remove_annotations:
        image = _remove_bottom_annotations(image, background=background)
    if auto_crop:
        image = _crop_to_content(image, background=background)

    converted = _fit_on_square(
        image,
        size=size,
        background=background,
        margin=margin,
    )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    converted.save(destination_path, format="PNG", optimize=True)
    return destination_path


def convert_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    size: int = OUTPUT_SIZE,
    margin: int = 10,
    auto_crop: bool = True,
    remove_annotations: bool = True,
) -> list[Path]:
    """Convert every PNG directly inside input_dir and return output paths."""
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {source_dir}")

    source_paths = sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        ),
        key=lambda path: path.name.lower(),
    )
    if not source_paths:
        raise FileNotFoundError(f"No PNG files found in: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for index, source_path in enumerate(source_paths, start=1):
        output_path = destination_dir / source_path.name
        convert_image(
            source_path,
            output_path,
            size=size,
            margin=margin,
            auto_crop=auto_crop,
            remove_annotations=remove_annotations,
        )
        output_paths.append(output_path)
        print(f"[{index}/{len(source_paths)}] Saved {output_path.name}")

    return output_paths


def _is_conforming_png(path: Path, *, size: int) -> bool:
    """Return whether a PNG is already an RGB square of the requested size."""
    with Image.open(path) as image:
        return (
            image.format == "PNG"
            and image.size == (size, size)
            and image.mode == "RGB"
        )


def convert_nonconforming_in_place(
    input_dir: str | Path,
    *,
    size: int = OUTPUT_SIZE,
    margin: int = 10,
    auto_crop: bool = True,
    remove_annotations: bool = True,
    recursive: bool = True,
) -> list[Path]:
    """Find non-conforming PNGs and overwrite them with converted images.

    Existing RGB PNGs of the requested size are left untouched. Images that
    need conversion are resized without changing their aspect ratio, centered
    on a square background, converted to RGB, and saved back to the same path.
    """
    source_dir = Path(input_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {source_dir}"
        )

    candidates = (
        source_dir.rglob("*.png")
        if recursive
        else source_dir.glob("*.png")
    )
    source_paths = sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: str(path).lower(),
    )
    if not source_paths:
        raise FileNotFoundError(f"No PNG files found in: {source_dir}")

    paths_to_convert = [
        path
        for path in source_paths
        if not _is_conforming_png(path, size=size)
    ]
    if not paths_to_convert:
        print(
            f"All {len(source_paths)} PNG files already conform to "
            f"{size}x{size} RGB."
        )
        return []

    converted_paths: list[Path] = []
    for index, source_path in enumerate(paths_to_convert, start=1):
        convert_image(
            source_path,
            source_path,
            size=size,
            margin=margin,
            auto_crop=auto_crop,
            remove_annotations=remove_annotations,
        )
        converted_paths.append(source_path)
        print(
            f"[{index}/{len(paths_to_convert)}] Overwrote "
            f"{source_path}"
        )

    return converted_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PNG files to square RGB images. By default, recursively "
            "find non-conforming images under merged_data and overwrite them."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"source directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "optional separate destination directory; when omitted, only "
            "non-conforming files are converted in place"
        ),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=OUTPUT_SIZE,
        help=f"output width and height (default: {OUTPUT_SIZE})",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="margin around the resized diagram in pixels (default: 10)",
    )
    parser.add_argument(
        "--keep-full-image",
        action="store_true",
        help="disable automatic empty-background cropping",
    )
    parser.add_argument(
        "--keep-annotations",
        action="store_true",
        help="keep the explanatory text below each Lewis structure",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="when converting in place, inspect only the input directory itself",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    common_options = {
        "size": args.size,
        "margin": args.margin,
        "auto_crop": not args.keep_full_image,
        "remove_annotations": not args.keep_annotations,
    }
    if args.output_dir is None:
        output_paths = convert_nonconforming_in_place(
            args.input_dir,
            recursive=not args.non_recursive,
            **common_options,
        )
        print(
            f"Converted and overwrote {len(output_paths)} non-conforming "
            f"PNG files as {args.size}x{args.size} RGB images."
        )
    else:
        output_paths = convert_directory(
            args.input_dir,
            args.output_dir,
            **common_options,
        )
        print(
            f"Converted {len(output_paths)} PNG files to "
            f"{args.size}x{args.size} RGB images in {args.output_dir}"
        )


if __name__ == "__main__":
    main()
