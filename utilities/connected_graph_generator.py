r"""Generate distinct connected-graph training images with NetworkX.

Default usage:

    .\.venv\Scripts\python.exe utilities\connected_graph_generator.py

The command creates 150 RGB 224 x 224 PNG files in
``data_Connected_graph\generated``. Each graph is connected and
non-isomorphic to every other generated graph. No class labels or captions
are drawn into the images.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_Connected_graph" / "generated"
DEFAULT_COUNT = 150
DEFAULT_SIZE = 224
DEFAULT_SEED = 20260729
SUPERSAMPLE = 3
STYLE_NAMES = ("standard", "handwritten")


@dataclass(frozen=True)
class GraphSample:
    family: str
    graph: nx.Graph


@dataclass(frozen=True)
class RenderTheme:
    background: tuple[int, int, int]
    edge: tuple[int, int, int]
    node_fill: tuple[int, int, int]
    node_outline: tuple[int, int, int]


STANDARD_THEMES: tuple[RenderTheme, ...] = (
    RenderTheme((255, 255, 255), (55, 65, 81), (225, 238, 255), (35, 105, 190)),
    RenderTheme((250, 252, 255), (60, 70, 82), (239, 245, 250), (31, 41, 55)),
    RenderTheme((247, 250, 247), (55, 71, 61), (221, 244, 228), (34, 123, 73)),
    RenderTheme((18, 24, 33), (116, 143, 180), (29, 78, 137), (113, 182, 255)),
)

HANDWRITTEN_THEMES: tuple[RenderTheme, ...] = (
    RenderTheme((247, 244, 232), (47, 52, 56), (247, 244, 232), (38, 42, 46)),
    RenderTheme((250, 248, 237), (28, 74, 139), (250, 248, 237), (24, 68, 132)),
    RenderTheme((239, 239, 235), (70, 70, 67), (239, 239, 235), (60, 60, 58)),
)


def _relabel(graph: nx.Graph) -> nx.Graph:
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def _graph_uniqueness_key(graph: nx.Graph) -> tuple[object, ...]:
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        tuple(sorted(dict(graph.degree()).values())),
    )


def _classic_graphs() -> list[nx.Graph]:
    return [
        nx.bull_graph(),
        nx.chvatal_graph(),
        nx.cubical_graph(),
        nx.desargues_graph(),
        nx.diamond_graph(),
        nx.dodecahedral_graph(),
        nx.frucht_graph(),
        nx.heawood_graph(),
        nx.house_graph(),
        nx.house_x_graph(),
        nx.icosahedral_graph(),
        nx.krackhardt_kite_graph(),
        nx.moebius_kantor_graph(),
        nx.octahedral_graph(),
        nx.pappus_graph(),
        nx.petersen_graph(),
    ]


def _random_connected_graph(
    node_count: int,
    *,
    seed: int,
    extra_edge_count: int,
) -> nx.Graph:
    rng = random.Random(seed)
    graph = nx.random_labeled_tree(node_count, seed=seed)
    possible_edges = list(nx.non_edges(graph))
    rng.shuffle(possible_edges)
    graph.add_edges_from(possible_edges[:extra_edge_count])
    return graph


def _candidate_families(seed: int) -> dict[str, list[nx.Graph]]:
    rng = random.Random(seed)
    families: dict[str, list[nx.Graph]] = {
        "path": [nx.path_graph(node_count) for node_count in range(3, 15)],
        "cycle": [nx.cycle_graph(node_count) for node_count in range(3, 15)],
        "star": [nx.star_graph(leaf_count) for leaf_count in range(3, 15)],
        "wheel": [nx.wheel_graph(node_count) for node_count in range(4, 16)],
        "complete": [
            nx.complete_graph(node_count) for node_count in range(3, 13)
        ],
        "complete-bipartite": [
            nx.complete_bipartite_graph(left_count, right_count)
            for left_count, right_count in (
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (2, 7),
                (3, 3),
                (3, 4),
                (3, 5),
                (3, 6),
                (4, 4),
                (4, 5),
            )
        ],
        "ladder": [nx.ladder_graph(length) for length in range(2, 12)],
        "circular-ladder": [
            nx.circular_ladder_graph(length) for length in range(3, 13)
        ],
        "barbell": [
            nx.barbell_graph(clique_size, bridge_size)
            for clique_size in range(3, 9)
            for bridge_size in (0, 1)
        ],
        "lollipop": [
            nx.lollipop_graph(clique_size, path_length)
            for clique_size in range(3, 9)
            for path_length in (2, 3)
        ],
        "rary-tree": [
            nx.full_rary_tree(branching, node_count)
            for branching, node_count in (
                (2, 7),
                (2, 9),
                (2, 11),
                (2, 13),
                (2, 15),
                (2, 17),
                (3, 8),
                (3, 10),
                (3, 13),
                (3, 16),
                (4, 9),
                (4, 13),
            )
        ],
        "grid": [
            nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, columns))
            for rows, columns in (
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (2, 7),
                (2, 8),
                (2, 9),
                (2, 10),
                (3, 3),
                (3, 4),
                (3, 5),
                (3, 6),
            )
        ],
        "classic": _classic_graphs(),
        "random-tree": [
            nx.random_labeled_tree(node_count, seed=rng.randrange(2**32))
            for node_count in range(8, 20)
        ],
        "random-connected": [
            _random_connected_graph(
                node_count=7 + index % 12,
                seed=rng.randrange(2**32),
                extra_edge_count=1 + index % 10,
            )
            for index in range(30)
        ],
    }
    return families


def build_graph_catalog(
    *,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[GraphSample]:
    """Build connected, pairwise non-isomorphic graphs."""
    if count < 1:
        raise ValueError("count must be greater than zero")

    rng = random.Random(seed)
    families = _candidate_families(seed)
    family_names = list(families)
    for candidates in families.values():
        rng.shuffle(candidates)
    rng.shuffle(family_names)

    uniqueness_buckets: dict[tuple[object, ...], list[nx.Graph]] = defaultdict(list)
    samples: list[GraphSample] = []

    def add_candidate(family: str, candidate: nx.Graph) -> bool:
        graph = _relabel(candidate)
        if graph.number_of_nodes() < 3 or not nx.is_connected(graph):
            return False

        key = _graph_uniqueness_key(graph)
        if any(nx.is_isomorphic(graph, existing) for existing in uniqueness_buckets[key]):
            return False

        uniqueness_buckets[key].append(graph)
        samples.append(GraphSample(family=family, graph=graph))
        return True

    while len(samples) < count and any(families.values()):
        for family in family_names:
            if not families[family]:
                continue
            add_candidate(family, families[family].pop())
            if len(samples) == count:
                break

    attempt = 0
    while len(samples) < count and attempt < 10_000:
        attempt += 1
        node_count = rng.randint(7, 22)
        graph = _random_connected_graph(
            node_count,
            seed=rng.randrange(2**32),
            extra_edge_count=rng.randint(1, max(2, node_count * 2)),
        )
        add_candidate("random-connected", graph)

    if len(samples) != count:
        raise RuntimeError(f"Could only construct {len(samples)} unique graphs")
    return samples


def _layout_graph(
    graph: nx.Graph,
    *,
    family: str,
    seed: int,
    size: int,
    margin: float,
    handwritten: bool,
) -> dict[int, tuple[float, float]]:
    rng = random.Random(seed)

    if family in {"cycle", "wheel", "complete", "circular-ladder"}:
        raw_positions = nx.circular_layout(graph)
    elif family == "complete-bipartite":
        top_nodes = {
            node
            for node, attributes in graph.nodes(data=True)
            if attributes.get("bipartite") == 0
        }
        raw_positions = nx.bipartite_layout(graph, top_nodes, align="vertical")
    elif family == "star":
        center = max(graph, key=graph.degree)
        leaves = [node for node in graph if node != center]
        raw_positions = {center: np.array((0.0, 0.0))}
        for index, node in enumerate(leaves):
            angle = 2 * math.pi * index / len(leaves)
            raw_positions[node] = np.array((math.cos(angle), math.sin(angle)))
    else:
        iterations = 120 if graph.number_of_nodes() <= 20 else 80
        raw_positions = nx.spring_layout(
            graph,
            seed=seed,
            iterations=iterations,
            k=1.15 / math.sqrt(graph.number_of_nodes()),
        )

    rotation_range = 9 if handwritten else 28
    rotation = math.radians(rng.uniform(-rotation_range, rotation_range))
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    rotated = {
        node: np.array(
            (
                position[0] * cosine - position[1] * sine,
                position[0] * sine + position[1] * cosine,
            )
        )
        for node, position in raw_positions.items()
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


def _draw_jittered_edge(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: tuple[int, int, int],
    width: int,
    jitter: float,
    rng: random.Random,
) -> None:
    if jitter <= 0:
        draw.line((start, end), fill=color, width=width)
        return

    points: list[tuple[float, float]] = []
    for index in range(7):
        fraction = index / 6
        x = start[0] + (end[0] - start[0]) * fraction
        y = start[1] + (end[1] - start[1]) * fraction
        if index not in (0, 6):
            x += rng.uniform(-jitter, jitter)
            y += rng.uniform(-jitter, jitter)
        points.append((x, y))
    draw.line(points, fill=color, width=width, joint="curve")


def _draw_handwritten_node(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    *,
    radius: float,
    theme: RenderTheme,
    width: int,
    rng: random.Random,
) -> None:
    points = []
    for index in range(20):
        angle = 2 * math.pi * index / 20
        local_radius = radius * rng.uniform(0.88, 1.12)
        points.append(
            (
                center[0] + math.cos(angle) * local_radius,
                center[1] + math.sin(angle) * local_radius,
            )
        )
    draw.polygon(points, fill=theme.node_fill)
    draw.line([*points, points[0]], fill=theme.node_outline, width=width, joint="curve")


def _add_paper_texture(
    image: Image.Image,
    *,
    background: tuple[int, int, int],
    rng: random.Random,
    count: int,
) -> None:
    draw = ImageDraw.Draw(image)
    for _ in range(count):
        x = rng.randrange(image.width)
        y = rng.randrange(image.height)
        delta = rng.choice((-1, 1)) * rng.randint(2, 6)
        color = tuple(
            min(255, max(0, channel + delta)) for channel in background
        )
        radius = rng.choice((1, 1, 2))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def render_graph(
    sample: GraphSample,
    *,
    style: str,
    seed: int,
    size: int = DEFAULT_SIZE,
) -> Image.Image:
    """Render one graph as an RGB square PNG-ready image."""
    if style not in STYLE_NAMES:
        raise ValueError(f"Unknown style: {style}")
    if size < 64:
        raise ValueError("size must be at least 64")

    rng = random.Random(seed)
    handwritten = style == "handwritten"
    themes = HANDWRITTEN_THEMES if handwritten else STANDARD_THEMES
    theme = rng.choice(themes)
    scale = SUPERSAMPLE
    canvas_size = size * scale
    image = Image.new("RGB", (canvas_size, canvas_size), theme.background)
    if handwritten:
        _add_paper_texture(
            image,
            background=theme.background,
            rng=rng,
            count=320,
        )
    draw = ImageDraw.Draw(image)

    node_count = sample.graph.number_of_nodes()
    if node_count <= 8:
        logical_radius = rng.uniform(7.0, 8.5)
    elif node_count <= 16:
        logical_radius = rng.uniform(5.3, 6.8)
    else:
        logical_radius = rng.uniform(4.0, 5.2)
    logical_margin = logical_radius + 17

    logical_positions = _layout_graph(
        sample.graph,
        family=sample.family,
        seed=seed,
        size=size,
        margin=logical_margin,
        handwritten=handwritten,
    )
    positions = {
        node: (position[0] * scale, position[1] * scale)
        for node, position in logical_positions.items()
    }

    edge_width = max(
        2,
        round((1.5 if handwritten else 1.8) * scale),
    )
    edge_jitter = (1.1 * scale) if handwritten else 0.0
    for start_node, end_node in sample.graph.edges():
        _draw_jittered_edge(
            draw,
            positions[start_node],
            positions[end_node],
            color=theme.edge,
            width=edge_width,
            jitter=edge_jitter,
            rng=rng,
        )

    radius = logical_radius * scale
    outline_width = max(2, round((1.4 if handwritten else 1.8) * scale))
    for center in positions.values():
        if handwritten:
            _draw_handwritten_node(
                draw,
                center,
                radius=radius,
                theme=theme,
                width=outline_width,
                rng=rng,
            )
        else:
            draw.ellipse(
                (
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius,
                ),
                fill=theme.node_fill,
                outline=theme.node_outline,
                width=outline_width,
            )

    if handwritten:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.12 * scale))
    return image.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")


def generate_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    count: int = DEFAULT_COUNT,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    """Generate distinct connected graphs and return their output paths."""
    if not 1 <= count <= DEFAULT_COUNT:
        raise ValueError(f"count must be between 1 and {DEFAULT_COUNT}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for old_path in destination.glob("connected-graph-*.png"):
        old_path.unlink()

    samples = build_graph_catalog(count=count, seed=seed)
    rng = random.Random(seed)
    output_paths: list[Path] = []

    for index, sample in enumerate(samples, start=1):
        style = rng.choice(STYLE_NAMES)
        image_seed = rng.randrange(2**32)
        image = render_graph(
            sample,
            style=style,
            seed=image_seed,
            size=size,
        )
        output_name = (
            f"connected-graph-{index:03d}-"
            f"{sample.family}-{style}.png"
        )
        output_path = destination / output_name
        image.save(output_path, format="PNG", optimize=True)
        output_paths.append(output_path)
        print(
            f"[{index}/{count}] Saved {output_path.name} "
            f"(nodes={sample.graph.number_of_nodes()}, "
            f"edges={sample.graph.number_of_edges()})"
        )

    family_counts = Counter(sample.family for sample in samples)
    print(f"Families: {dict(sorted(family_counts.items()))}")
    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate non-isomorphic connected graph PNG files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"number of graphs, at most {DEFAULT_COUNT}",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"output width and height (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random seed (default: {DEFAULT_SEED})",
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
        f"Generated {len(paths)} connected graph images in "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
