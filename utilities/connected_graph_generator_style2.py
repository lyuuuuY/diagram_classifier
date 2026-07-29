r"""Generate labeled, colored connected graphs with optional arrows.

Default usage:

    .\.venv\Scripts\python.exe utilities\connected_graph_generator_style2.py

The command creates 40 pairwise non-isomorphic RGB 224 x 224 PNG files in
``data_Connected_graph\generated_style2``. Nodes contain labels such as A,
B, C, may use different colors, and most images use directed arrow edges.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from PIL import Image, ImageDraw, ImageFont

try:
    from utilities.connected_graph_generator import (
        GraphSample,
        build_graph_catalog,
    )
except ModuleNotFoundError:
    from connected_graph_generator import GraphSample, build_graph_catalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data_Connected_graph" / "generated_style2"
)
DEFAULT_COUNT = 40
DEFAULT_SIZE = 224
DEFAULT_SEED = 20260730
SUPERSAMPLE = 3


@dataclass(frozen=True)
class DiagramTheme:
    name: str
    background: tuple[int, int, int]
    edge: tuple[int, int, int]
    outline: tuple[int, int, int]
    node_palette: tuple[tuple[int, int, int], ...]
    grid: tuple[int, int, int] | None = None


THEMES: tuple[DiagramTheme, ...] = (
    DiagramTheme(
        name="cool",
        background=(247, 250, 253),
        edge=(61, 77, 98),
        outline=(38, 65, 92),
        node_palette=(
            (116, 185, 255),
            (135, 225, 192),
            (255, 190, 116),
            (194, 157, 255),
            (255, 139, 157),
        ),
        grid=(228, 235, 241),
    ),
    DiagramTheme(
        name="paper",
        background=(255, 253, 247),
        edge=(65, 67, 71),
        outline=(52, 54, 57),
        node_palette=(
            (255, 221, 128),
            (167, 220, 194),
            (159, 197, 235),
            (238, 171, 182),
            (202, 184, 231),
        ),
    ),
    DiagramTheme(
        name="vivid",
        background=(250, 251, 250),
        edge=(50, 55, 60),
        outline=(35, 40, 45),
        node_palette=(
            (244, 96, 96),
            (255, 182, 72),
            (70, 190, 138),
            (73, 151, 232),
            (154, 105, 221),
        ),
    ),
    DiagramTheme(
        name="dark",
        background=(19, 27, 39),
        edge=(146, 168, 198),
        outline=(215, 229, 246),
        node_palette=(
            (48, 133, 214),
            (33, 171, 132),
            (211, 123, 43),
            (156, 91, 206),
            (205, 73, 107),
        ),
        grid=(32, 44, 60),
    ),
)


def _node_label(index: int) -> str:
    """Return spreadsheet-style labels: A...Z, AA, AB, and so on."""
    label = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _raw_layout(
    sample: GraphSample,
    *,
    seed: int,
) -> dict[int, tuple[float, float]]:
    graph = sample.graph
    family = sample.family

    if family in {
        "cycle",
        "wheel",
        "complete",
        "circular-ladder",
    }:
        positions = nx.circular_layout(graph)
    elif family == "star":
        positions = nx.shell_layout(
            graph,
            nlist=[
                [max(graph, key=graph.degree)],
                [node for node in graph if node != max(graph, key=graph.degree)],
            ],
        )
    elif family == "complete-bipartite":
        left_nodes = {
            node
            for node, attributes in graph.nodes(data=True)
            if attributes.get("bipartite") == 0
        }
        positions = nx.bipartite_layout(
            graph,
            left_nodes,
            align="vertical",
        )
    elif family in {"path", "ladder", "grid"}:
        positions = nx.spring_layout(
            graph,
            seed=seed,
            iterations=180,
            k=1.35 / math.sqrt(graph.number_of_nodes()),
        )
    else:
        positions = nx.spring_layout(
            graph,
            seed=seed,
            iterations=140,
            k=1.25 / math.sqrt(graph.number_of_nodes()),
        )

    return {
        int(node): (float(position[0]), float(position[1]))
        for node, position in positions.items()
    }


def _scale_layout(
    raw_positions: dict[int, tuple[float, float]],
    *,
    size: int,
    margin: float,
    seed: int,
) -> dict[int, tuple[float, float]]:
    rng = random.Random(seed)
    rotation = math.radians(rng.uniform(-18.0, 18.0))
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    rotated = {
        node: (
            x * cosine - y * sine,
            x * sine + y * cosine,
        )
        for node, (x, y) in raw_positions.items()
    }

    xs = [position[0] for position in rotated.values()]
    ys = [position[1] for position in rotated.values()]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    span_x = max(max(xs) - min(xs), 0.001)
    span_y = max(max(ys) - min(ys), 0.001)
    extent = max(span_x, span_y)
    usable_size = size - 2 * margin

    return {
        node: (
            size / 2 + (position[0] - center_x) / extent * usable_size,
            size / 2 + (position[1] - center_y) / extent * usable_size,
        )
        for node, position in rotated.items()
    }


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    *,
    size: int,
    color: tuple[int, int, int],
    scale: int,
) -> None:
    spacing = 22 * scale
    for coordinate in range(spacing, size, spacing):
        draw.line(
            (coordinate, 0, coordinate, size),
            fill=color,
            width=1,
        )
        draw.line(
            (0, coordinate, size, coordinate),
            fill=color,
            width=1,
        )


def _trim_edge(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    start_offset: float,
    end_offset: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = max(math.hypot(delta_x, delta_y), 0.001)
    unit = (delta_x / length, delta_y / length)
    trimmed_start = (
        start[0] + unit[0] * start_offset,
        start[1] + unit[1] * start_offset,
    )
    trimmed_end = (
        end[0] - unit[0] * end_offset,
        end[1] - unit[1] * end_offset,
    )
    return trimmed_start, trimmed_end, unit


def _draw_edge(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    node_radius: float,
    color: tuple[int, int, int],
    width: int,
    directed: bool,
    scale: int,
) -> None:
    arrow_length = 6.0 * scale
    start_offset = node_radius + 1.5 * scale
    end_offset = node_radius + 1.5 * scale
    edge_start, arrow_tip, unit = _trim_edge(
        start,
        end,
        start_offset=start_offset,
        end_offset=end_offset,
    )

    if not directed:
        draw.line(
            (edge_start, arrow_tip),
            fill=color,
            width=width,
        )
        return

    line_end = (
        arrow_tip[0] - unit[0] * arrow_length * 0.55,
        arrow_tip[1] - unit[1] * arrow_length * 0.55,
    )
    draw.line((edge_start, line_end), fill=color, width=width)

    perpendicular = (-unit[1], unit[0])
    arrow_base = (
        arrow_tip[0] - unit[0] * arrow_length,
        arrow_tip[1] - unit[1] * arrow_length,
    )
    arrow_half_width = 3.2 * scale
    arrow_left = (
        arrow_base[0] + perpendicular[0] * arrow_half_width,
        arrow_base[1] + perpendicular[1] * arrow_half_width,
    )
    arrow_right = (
        arrow_base[0] - perpendicular[0] * arrow_half_width,
        arrow_base[1] - perpendicular[1] * arrow_half_width,
    )
    draw.polygon(
        (arrow_tip, arrow_left, arrow_right),
        fill=color,
    )


def _text_color(fill: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = (
        0.2126 * fill[0]
        + 0.7152 * fill[1]
        + 0.0722 * fill[2]
    )
    return (20, 26, 31) if luminance > 155 else (255, 255, 255)


def render_style2_graph(
    sample: GraphSample,
    *,
    directed: bool,
    seed: int,
    size: int = DEFAULT_SIZE,
) -> tuple[Image.Image, str]:
    """Render one labeled graph and return the image and theme name."""
    if size < 96:
        raise ValueError("size must be at least 96")

    rng = random.Random(seed)
    theme = rng.choice(THEMES)
    scale = SUPERSAMPLE
    canvas_size = size * scale
    image = Image.new(
        "RGB",
        (canvas_size, canvas_size),
        theme.background,
    )
    draw = ImageDraw.Draw(image)
    if theme.grid is not None and rng.random() < 0.65:
        _draw_grid(
            draw,
            size=canvas_size,
            color=theme.grid,
            scale=scale,
        )

    node_count = sample.graph.number_of_nodes()
    if node_count <= 7:
        logical_radius = 13.0
        logical_font_size = 12
    elif node_count <= 13:
        logical_radius = 10.5
        logical_font_size = 9
    else:
        logical_radius = 8.2
        logical_font_size = 7

    raw_positions = _raw_layout(sample, seed=seed)
    logical_positions = _scale_layout(
        raw_positions,
        size=size,
        margin=logical_radius + 13,
        seed=seed,
    )
    positions = {
        node: (x * scale, y * scale)
        for node, (x, y) in logical_positions.items()
    }
    radius = logical_radius * scale

    edges = list(sample.graph.edges())
    rng.shuffle(edges)
    edge_width = max(3, round(1.55 * scale))
    for start_node, end_node in edges:
        if directed and rng.random() < 0.5:
            start_node, end_node = end_node, start_node
        _draw_edge(
            draw,
            positions[start_node],
            positions[end_node],
            node_radius=radius,
            color=theme.edge,
            width=edge_width,
            directed=directed,
            scale=scale,
        )

    palette = list(theme.node_palette)
    rng.shuffle(palette)
    font = _font(logical_font_size * scale)
    outline_width = max(3, round(1.5 * scale))
    shadow_offset = 1.6 * scale
    for node in sorted(sample.graph.nodes()):
        center_x, center_y = positions[node]
        fill = palette[node % len(palette)]
        draw.ellipse(
            (
                center_x - radius + shadow_offset,
                center_y - radius + shadow_offset,
                center_x + radius + shadow_offset,
                center_y + radius + shadow_offset,
            ),
            fill=(0, 0, 0),
        )
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=fill,
            outline=theme.outline,
            width=outline_width,
        )
        draw.text(
            (center_x, center_y),
            _node_label(node),
            fill=_text_color(fill),
            font=font,
            anchor="mm",
            stroke_width=0,
        )

    return (
        image.resize(
            (size, size),
            Image.Resampling.LANCZOS,
        ).convert("RGB"),
        theme.name,
    )


def generate_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    count: int = DEFAULT_COUNT,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    """Generate style2 images and return their output paths."""
    if not 1 <= count <= DEFAULT_COUNT:
        raise ValueError(f"count must be between 1 and {DEFAULT_COUNT}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for old_path in destination.glob("connected-graph-style2-*.png"):
        old_path.unlink()

    samples = build_graph_catalog(count=count, seed=seed)
    rng = random.Random(seed)
    output_paths: list[Path] = []
    direction_counts: Counter[str] = Counter()
    theme_counts: Counter[str] = Counter()

    for index, sample in enumerate(samples, start=1):
        directed = index % 3 != 0
        image_seed = rng.randrange(2**32)
        image, theme_name = render_style2_graph(
            sample,
            directed=directed,
            seed=image_seed,
            size=size,
        )
        direction_name = "directed" if directed else "undirected"
        output_name = (
            f"connected-graph-style2-{index:03d}-"
            f"{sample.family}-{direction_name}-{theme_name}.png"
        )
        output_path = destination / output_name
        image.save(output_path, format="PNG", optimize=True)
        output_paths.append(output_path)
        direction_counts[direction_name] += 1
        theme_counts[theme_name] += 1
        print(
            f"[{index}/{count}] Saved {output_path.name} "
            f"(nodes={sample.graph.number_of_nodes()}, "
            f"edges={sample.graph.number_of_edges()})"
        )

    print(f"Directions: {dict(direction_counts)}")
    print(f"Themes: {dict(theme_counts)}")
    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate labeled colored connected graphs with optional arrows."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = generate_dataset(
        args.output_dir,
        count=args.count,
        size=args.size,
        seed=args.seed,
    )
    print(
        f"Generated {len(paths)} style2 graph images in "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
