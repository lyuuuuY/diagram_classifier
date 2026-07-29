r"""Generate synthetic Lewis-structure training images with Pillow.

The default command creates 130 unique RGB PNG files in
``data_lewis\generated``:

    .\.venv\Scripts\python.exe utilities\lewis_generator.py

No explanatory captions or class labels are drawn into the images.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_lewis" / "generated"
DEFAULT_COUNT = 130
DEFAULT_SIZE = 224
DEFAULT_SEED = 20260729
SUPERSAMPLE = 3


@dataclass(frozen=True)
class Atom:
    symbol: str
    x: float
    y: float
    lone_pair_angles: tuple[float, ...] = ()
    formal_charge: str | None = None


@dataclass(frozen=True)
class Bond:
    start: int
    end: int
    order: int = 1


@dataclass(frozen=True)
class Molecule:
    slug: str
    atoms: tuple[Atom, ...]
    bonds: tuple[Bond, ...]
    overall_charge: str | None = None


@dataclass(frozen=True)
class RenderStyle:
    name: str
    background: tuple[int, int, int]
    atom_color: tuple[int, int, int]
    bond_color: tuple[int, int, int]
    electron_color: tuple[int, int, int]
    font_names: tuple[str, ...]
    line_width: float
    jitter: float
    rotation: float
    paper_specks: int = 0
    blur_radius: float = 0.0


STYLES: tuple[RenderStyle, ...] = (
    RenderStyle(
        name="clean",
        background=(255, 255, 252),
        atom_color=(24, 28, 34),
        bond_color=(35, 40, 48),
        electron_color=(24, 28, 34),
        font_names=("arialbd.ttf", "DejaVuSans-Bold.ttf"),
        line_width=2.4,
        jitter=0.0,
        rotation=1.5,
    ),
    RenderStyle(
        name="dark",
        background=(16, 23, 32),
        atom_color=(242, 246, 251),
        bond_color=(113, 174, 255),
        electron_color=(45, 132, 238),
        font_names=("arialbd.ttf", "DejaVuSans-Bold.ttf"),
        line_width=2.3,
        jitter=0.0,
        rotation=1.0,
    ),
    RenderStyle(
        name="handwritten",
        background=(239, 239, 235),
        atom_color=(65, 65, 63),
        bond_color=(76, 76, 73),
        electron_color=(65, 65, 63),
        font_names=("comic.ttf", "Inkfree.ttf", "arial.ttf"),
        line_width=1.8,
        jitter=1.0,
        rotation=5.5,
        paper_specks=130,
        blur_radius=0.25,
    ),
)


HALOGENS = {"F", "Cl", "Br", "I"}


def _terminal_lone_pair_count(symbol: str, bond_order: int) -> int:
    if symbol == "H":
        return 0
    if symbol in HALOGENS:
        return 3
    if symbol in {"O", "S", "Se", "Te"}:
        return 2 if bond_order >= 2 else 3
    if symbol == "N":
        return 1 if bond_order == 3 else 2
    return 0


def _pair_angles(outward_angle: float, count: int) -> tuple[float, ...]:
    if count == 0:
        return ()
    if count == 1:
        return (outward_angle,)
    if count == 2:
        return (outward_angle - 90, outward_angle + 90)
    if count == 3:
        return (outward_angle, outward_angle - 90, outward_angle + 90)
    raise ValueError(f"Unsupported lone-pair count: {count}")


def _diatomic(
    slug: str,
    left_symbol: str,
    right_symbol: str,
    *,
    bond_order: int = 1,
    left_pairs: int = 0,
    right_pairs: int = 0,
    left_charge: str | None = None,
    right_charge: str | None = None,
) -> Molecule:
    return Molecule(
        slug=slug,
        atoms=(
            Atom(
                left_symbol,
                -0.72,
                0.0,
                _pair_angles(180, left_pairs),
                left_charge,
            ),
            Atom(
                right_symbol,
                0.72,
                0.0,
                _pair_angles(0, right_pairs),
                right_charge,
            ),
        ),
        bonds=(Bond(0, 1, bond_order),),
    )


def _radial(
    slug: str,
    center_symbol: str,
    terminal_symbols: Sequence[str],
    terminal_angles: Sequence[float],
    *,
    bond_orders: Sequence[int] | None = None,
    central_pairs: tuple[float, ...] = (),
    terminal_pair_counts: Sequence[int] | None = None,
    center_charge: str | None = None,
    terminal_charges: Sequence[str | None] | None = None,
    overall_charge: str | None = None,
    radius: float = 1.0,
) -> Molecule:
    if len(terminal_symbols) != len(terminal_angles):
        raise ValueError("terminal symbols and angles must have equal lengths")

    orders = tuple(bond_orders or (1,) * len(terminal_symbols))
    if len(orders) != len(terminal_symbols):
        raise ValueError("one bond order is required for each terminal")

    if terminal_pair_counts is None:
        pair_counts = tuple(
            _terminal_lone_pair_count(symbol, order)
            for symbol, order in zip(terminal_symbols, orders)
        )
    else:
        pair_counts = tuple(terminal_pair_counts)

    charges = tuple(terminal_charges or (None,) * len(terminal_symbols))
    atoms = [Atom(center_symbol, 0.0, 0.0, central_pairs, center_charge)]
    for symbol, angle, pair_count, charge in zip(
        terminal_symbols,
        terminal_angles,
        pair_counts,
        charges,
    ):
        radians = math.radians(angle)
        atoms.append(
            Atom(
                symbol,
                math.cos(radians) * radius,
                math.sin(radians) * radius,
                _pair_angles(angle, pair_count),
                charge,
            )
        )

    return Molecule(
        slug=slug,
        atoms=tuple(atoms),
        bonds=tuple(
            Bond(0, terminal_index, order)
            for terminal_index, order in enumerate(orders, start=1)
        ),
        overall_charge=overall_charge,
    )


def _hydrocarbon(slug: str, carbon_bond_orders: Sequence[int]) -> Molecule:
    """Build an explicit two- or three-carbon hydrocarbon."""
    carbon_count = len(carbon_bond_orders) + 1
    carbon_spacing = 1.15
    carbon_x = [
        (index - (carbon_count - 1) / 2) * carbon_spacing
        for index in range(carbon_count)
    ]
    atoms: list[Atom] = [Atom("C", x, 0.0) for x in carbon_x]
    bonds: list[Bond] = [
        Bond(index, index + 1, order)
        for index, order in enumerate(carbon_bond_orders)
    ]

    for carbon_index in range(carbon_count):
        used_valence = 0
        if carbon_index > 0:
            used_valence += carbon_bond_orders[carbon_index - 1]
        if carbon_index < carbon_count - 1:
            used_valence += carbon_bond_orders[carbon_index]
        hydrogen_count = 4 - used_valence

        if carbon_index == 0:
            angle_options = {
                1: (180,),
                2: (140, 220),
                3: (180, -90, 90),
            }
        elif carbon_index == carbon_count - 1:
            angle_options = {
                1: (0,),
                2: (-40, 40),
                3: (0, -90, 90),
            }
        else:
            angle_options = {
                0: (),
                1: (-90,),
                2: (-90, 90),
            }

        for angle in angle_options[hydrogen_count]:
            radians = math.radians(angle)
            hydrogen_index = len(atoms)
            atoms.append(
                Atom(
                    "H",
                    carbon_x[carbon_index] + math.cos(radians) * 0.72,
                    math.sin(radians) * 0.72,
                )
            )
            bonds.append(Bond(carbon_index, hydrogen_index))

    return Molecule(slug=slug, atoms=tuple(atoms), bonds=tuple(bonds))


def _organic_molecules() -> tuple[Molecule, ...]:
    methanol = Molecule(
        slug="methanol",
        atoms=(
            Atom("C", -0.55, 0.0),
            Atom("O", 0.55, 0.0, (-90, 90)),
            Atom("H", 1.42, 0.0),
            Atom("H", -1.27, 0.0),
            Atom("H", -0.55, -0.72),
            Atom("H", -0.55, 0.72),
        ),
        bonds=(
            Bond(0, 1),
            Bond(1, 2),
            Bond(0, 3),
            Bond(0, 4),
            Bond(0, 5),
        ),
    )
    ethanol = Molecule(
        slug="ethanol",
        atoms=(
            Atom("C", -1.05, 0.0),
            Atom("C", 0.0, 0.0),
            Atom("O", 1.05, 0.0, (-90, 90)),
            Atom("H", 1.88, 0.0),
            Atom("H", -1.77, 0.0),
            Atom("H", -1.05, -0.72),
            Atom("H", -1.05, 0.72),
            Atom("H", 0.0, -0.72),
            Atom("H", 0.0, 0.72),
        ),
        bonds=(
            Bond(0, 1),
            Bond(1, 2),
            Bond(2, 3),
            Bond(0, 4),
            Bond(0, 5),
            Bond(0, 6),
            Bond(1, 7),
            Bond(1, 8),
        ),
    )
    dimethyl_ether = Molecule(
        slug="dimethyl-ether",
        atoms=(
            Atom("C", -1.10, 0.0),
            Atom("O", 0.0, 0.0, (-90, 90)),
            Atom("C", 1.10, 0.0),
            Atom("H", -1.82, 0.0),
            Atom("H", -1.10, -0.72),
            Atom("H", -1.10, 0.72),
            Atom("H", 1.82, 0.0),
            Atom("H", 1.10, -0.72),
            Atom("H", 1.10, 0.72),
        ),
        bonds=(
            Bond(0, 1),
            Bond(1, 2),
            Bond(0, 3),
            Bond(0, 4),
            Bond(0, 5),
            Bond(2, 6),
            Bond(2, 7),
            Bond(2, 8),
        ),
    )
    formaldehyde = Molecule(
        slug="formaldehyde",
        atoms=(
            Atom("C", 0.0, 0.0),
            Atom("O", 1.05, 0.0, (-90, 90)),
            Atom("H", -0.78, -0.58),
            Atom("H", -0.78, 0.58),
        ),
        bonds=(Bond(0, 1, 2), Bond(0, 2), Bond(0, 3)),
    )
    formic_acid = Molecule(
        slug="formic-acid",
        atoms=(
            Atom("C", 0.0, 0.0),
            Atom("O", 0.0, -1.0, (-160, -20)),
            Atom("O", 0.88, 0.55, (-35, 55)),
            Atom("H", 1.70, 0.55),
            Atom("H", -0.88, 0.55),
        ),
        bonds=(
            Bond(0, 1, 2),
            Bond(0, 2),
            Bond(2, 3),
            Bond(0, 4),
        ),
    )
    acetic_acid = Molecule(
        slug="acetic-acid",
        atoms=(
            Atom("C", -0.85, 0.0),
            Atom("C", 0.25, 0.0),
            Atom("O", 0.25, -1.0, (-160, -20)),
            Atom("O", 1.12, 0.55, (-35, 55)),
            Atom("H", 1.90, 0.55),
            Atom("H", -1.57, 0.0),
            Atom("H", -0.85, -0.72),
            Atom("H", -0.85, 0.72),
        ),
        bonds=(
            Bond(0, 1),
            Bond(1, 2, 2),
            Bond(1, 3),
            Bond(3, 4),
            Bond(0, 5),
            Bond(0, 6),
            Bond(0, 7),
        ),
    )
    return (
        _hydrocarbon("ethane", (1,)),
        _hydrocarbon("ethylene", (2,)),
        _hydrocarbon("acetylene", (3,)),
        _hydrocarbon("propane", (1, 1)),
        _hydrocarbon("propene", (2, 1)),
        _hydrocarbon("propyne", (3, 1)),
        _hydrocarbon("allene", (2, 2)),
        methanol,
        ethanol,
        dimethyl_ether,
        formaldehyde,
        formic_acid,
        acetic_acid,
    )


def _molecules() -> tuple[Molecule, ...]:
    """Build a catalog of exactly 130 distinct Lewis structures."""
    molecules: list[Molecule] = []

    diatomic_specs = (
        ("hydrogen", "H", "H", 1, 0, 0, None, None),
        ("nitrogen", "N", "N", 3, 1, 1, None, None),
        ("oxygen", "O", "O", 2, 2, 2, None, None),
        ("fluorine", "F", "F", 1, 3, 3, None, None),
        ("chlorine", "Cl", "Cl", 1, 3, 3, None, None),
        ("bromine", "Br", "Br", 1, 3, 3, None, None),
        ("iodine", "I", "I", 1, 3, 3, None, None),
        ("carbon-monoxide", "C", "O", 3, 1, 1, "-", "+"),
        ("hydrogen-fluoride", "H", "F", 1, 0, 3, None, None),
        ("hydrogen-chloride", "H", "Cl", 1, 0, 3, None, None),
        ("hydrogen-bromide", "H", "Br", 1, 0, 3, None, None),
        ("hydrogen-iodide", "H", "I", 1, 0, 3, None, None),
        ("chlorine-monofluoride", "Cl", "F", 1, 3, 3, None, None),
        ("bromine-monofluoride", "Br", "F", 1, 3, 3, None, None),
        ("iodine-monofluoride", "I", "F", 1, 3, 3, None, None),
        ("iodine-monochloride", "I", "Cl", 1, 3, 3, None, None),
        ("iodine-monobromide", "I", "Br", 1, 3, 3, None, None),
        ("bromine-monochloride", "Br", "Cl", 1, 3, 3, None, None),
    )
    molecules.extend(
        _diatomic(
            slug,
            left,
            right,
            bond_order=order,
            left_pairs=left_pairs,
            right_pairs=right_pairs,
            left_charge=left_charge,
            right_charge=right_charge,
        )
        for (
            slug,
            left,
            right,
            order,
            left_pairs,
            right_pairs,
            left_charge,
            right_charge,
        ) in diatomic_specs
    )

    bent_angles = (35, 145)
    linear_angles = (180, 0)
    for slug, center in (
        ("water", "O"),
        ("hydrogen-sulfide", "S"),
        ("hydrogen-selenide", "Se"),
        ("hydrogen-telluride", "Te"),
    ):
        molecules.append(
            _radial(
                slug,
                center,
                ("H", "H"),
                bent_angles,
                central_pairs=(-120, -60),
            )
        )

    molecules.extend(
        (
            _radial(
                "carbon-dioxide",
                "C",
                ("O", "O"),
                linear_angles,
                bond_orders=(2, 2),
            ),
            _radial(
                "carbon-disulfide",
                "C",
                ("S", "S"),
                linear_angles,
                bond_orders=(2, 2),
            ),
            _radial(
                "carbonyl-sulfide",
                "C",
                ("O", "S"),
                linear_angles,
                bond_orders=(2, 2),
            ),
        )
    )

    for halogen in ("F", "Cl", "Br", "I"):
        molecules.append(
            _radial(
                f"beryllium-{halogen.lower()}2",
                "Be",
                (halogen, halogen),
                linear_angles,
            )
        )

    for center, names in (
        (
            "O",
            (
                ("oxygen-difluoride", "F"),
                ("dichlorine-monoxide", "Cl"),
                ("dibromine-monoxide", "Br"),
            ),
        ),
        (
            "S",
            (
                ("sulfur-difluoride", "F"),
                ("sulfur-dichloride", "Cl"),
                ("sulfur-dibromide", "Br"),
                ("sulfur-diiodide", "I"),
            ),
        ),
    ):
        for slug, terminal in names:
            molecules.append(
                _radial(
                    slug,
                    center,
                    (terminal, terminal),
                    bent_angles,
                    central_pairs=(-120, -60),
                )
            )

    molecules.extend(
        (
            _radial(
                "xenon-difluoride",
                "Xe",
                ("F", "F"),
                linear_angles,
                central_pairs=(-90, 90, 45),
            ),
            _radial(
                "triiodide-ion",
                "I",
                ("I", "I"),
                linear_angles,
                central_pairs=(-90, 90, 45),
                overall_charge="-",
            ),
            _radial(
                "sulfur-dioxide",
                "S",
                ("O", "O"),
                bent_angles,
                bond_orders=(2, 2),
                central_pairs=(-90,),
            ),
            _radial(
                "ozone",
                "O",
                ("O", "O"),
                bent_angles,
                bond_orders=(1, 2),
                central_pairs=(-90,),
                center_charge="+",
                terminal_charges=("-", None),
            ),
            _radial(
                "nitrite-ion",
                "N",
                ("O", "O"),
                bent_angles,
                bond_orders=(1, 2),
                central_pairs=(-90,),
                terminal_charges=("-", None),
                overall_charge="-",
            ),
            _radial(
                "nitronium-ion",
                "N",
                ("O", "O"),
                linear_angles,
                bond_orders=(2, 2),
                overall_charge="+",
            ),
            _radial(
                "hydrogen-cyanide",
                "C",
                ("H", "N"),
                linear_angles,
                bond_orders=(1, 3),
            ),
        )
    )

    planar_angles = (-90, 30, 150)
    pyramidal_angles = (30, 90, 150)
    for center in ("B", "Al"):
        for terminal in ("F", "Cl", "Br", "I"):
            molecules.append(
                _radial(
                    f"{center.lower()}-{terminal.lower()}3",
                    center,
                    (terminal,) * 3,
                    planar_angles,
                )
            )
    for center in ("N", "P", "As", "Sb"):
        for terminal in ("H", "F", "Cl", "Br", "I"):
            molecules.append(
                _radial(
                    f"{center.lower()}-{terminal.lower()}3",
                    center,
                    (terminal,) * 3,
                    pyramidal_angles,
                    central_pairs=(-90,),
                )
            )

    tetrahedral_angles = (-90, 0, 90, 180)
    for center in ("C", "Si", "Ge", "Sn"):
        for terminal in ("H", "F", "Cl", "Br", "I"):
            molecules.append(
                _radial(
                    f"{center.lower()}-{terminal.lower()}4",
                    center,
                    (terminal,) * 4,
                    tetrahedral_angles,
                )
            )
    molecules.extend(
        (
            _radial(
                "ammonium-ion",
                "N",
                ("H",) * 4,
                tetrahedral_angles,
                overall_charge="+",
            ),
            _radial(
                "phosphonium-ion",
                "P",
                ("H",) * 4,
                tetrahedral_angles,
                overall_charge="+",
            ),
            _radial(
                "tetrafluoroborate-ion",
                "B",
                ("F",) * 4,
                tetrahedral_angles,
                overall_charge="-",
            ),
            _radial(
                "tetrachloroaluminate-ion",
                "Al",
                ("Cl",) * 4,
                tetrahedral_angles,
                overall_charge="-",
            ),
        )
    )

    five_angles = (-90, -18, 54, 126, 198)
    for center in ("P", "As", "Sb"):
        for terminal in ("F", "Cl", "Br"):
            molecules.append(
                _radial(
                    f"{center.lower()}-{terminal.lower()}5",
                    center,
                    (terminal,) * 5,
                    five_angles,
                )
            )
    for center in ("I", "Br", "Cl"):
        molecules.append(
            _radial(
                f"{center.lower()}-f5",
                center,
                ("F",) * 5,
                five_angles,
                central_pairs=(90,),
            )
        )

    six_angles = (-90, -30, 30, 90, 150, 210)
    for center in ("S", "Se", "Te", "W", "Mo", "U", "Re", "Pt"):
        molecules.append(
            _radial(
                f"{center.lower()}-f6",
                center,
                ("F",) * 6,
                six_angles,
            )
        )
    molecules.extend(
        (
            _radial(
                "xenon-hexafluoride",
                "Xe",
                ("F",) * 6,
                six_angles,
                central_pairs=(0,),
            ),
            _radial(
                "hexafluorosilicate-ion",
                "Si",
                ("F",) * 6,
                six_angles,
                overall_charge="2-",
            ),
        )
    )

    molecules.extend(_organic_molecules())
    slugs = [molecule.slug for molecule in molecules]
    if len(molecules) != 130:
        raise RuntimeError(f"Expected 130 structures, built {len(molecules)}")
    if len(slugs) != len(set(slugs)):
        raise RuntimeError("Molecule slugs must be unique")
    return tuple(molecules)


def _font_search_paths(font_name: str) -> tuple[Path, ...]:
    return (
        Path("C:/Windows/Fonts") / font_name,
        Path("/usr/share/fonts/truetype/dejavu") / font_name,
        Path("/usr/local/share/fonts") / font_name,
    )


def _load_font(size: int, preferred_names: Sequence[str]) -> ImageFont.FreeTypeFont:
    for name in preferred_names:
        for path in _font_search_paths(name):
            if path.is_file():
                return ImageFont.truetype(str(path), size=size)

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def _draw_jittered_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    width: int,
    jitter: float,
    rng: random.Random,
) -> None:
    if jitter <= 0:
        draw.line((start, end), fill=fill, width=width)
        return

    points: list[tuple[float, float]] = []
    for step in range(6):
        fraction = step / 5
        x = start[0] + (end[0] - start[0]) * fraction
        y = start[1] + (end[1] - start[1]) * fraction
        if step not in (0, 5):
            x += rng.uniform(-jitter, jitter)
            y += rng.uniform(-jitter, jitter)
        points.append((x, y))
    draw.line(points, fill=fill, width=width, joint="curve")


def _draw_bond(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    order: int,
    style: RenderStyle,
    scale: int,
    rng: random.Random,
) -> None:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = math.hypot(delta_x, delta_y)
    if length == 0:
        return

    unit_x = delta_x / length
    unit_y = delta_y / length
    trim = min(12 * scale, length * 0.28)
    start_x = start[0] + unit_x * trim
    start_y = start[1] + unit_y * trim
    end_x = end[0] - unit_x * trim
    end_y = end[1] - unit_y * trim
    perpendicular_x = -unit_y
    perpendicular_y = unit_x

    if order == 1:
        offsets = (0.0,)
    elif order == 2:
        offsets = (-3.2 * scale, 3.2 * scale)
    elif order == 3:
        offsets = (-5.0 * scale, 0.0, 5.0 * scale)
    else:
        raise ValueError(f"Unsupported bond order: {order}")

    for offset in offsets:
        shifted_start = (
            start_x + perpendicular_x * offset,
            start_y + perpendicular_y * offset,
        )
        shifted_end = (
            end_x + perpendicular_x * offset,
            end_y + perpendicular_y * offset,
        )
        _draw_jittered_line(
            draw,
            shifted_start,
            shifted_end,
            fill=style.bond_color,
            width=max(1, round(style.line_width * scale)),
            jitter=style.jitter * scale,
            rng=rng,
        )


def _draw_lone_pair(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    angle_degrees: float,
    *,
    color: tuple[int, int, int],
    scale: int,
) -> None:
    angle = math.radians(angle_degrees)
    radial_x = math.cos(angle)
    radial_y = math.sin(angle)
    tangent_x = -radial_y
    tangent_y = radial_x
    radius = 22 * scale
    pair_separation = 3.3 * scale
    dot_radius = 1.8 * scale
    pair_center_x = center[0] + radial_x * radius
    pair_center_y = center[1] + radial_y * radius

    for direction in (-1, 1):
        dot_x = pair_center_x + tangent_x * pair_separation * direction
        dot_y = pair_center_y + tangent_y * pair_separation * direction
        draw.ellipse(
            (
                dot_x - dot_radius,
                dot_y - dot_radius,
                dot_x + dot_radius,
                dot_y + dot_radius,
            ),
            fill=color,
        )


def _draw_ion_brackets(
    draw: ImageDraw.ImageDraw,
    positions: Sequence[tuple[float, float]],
    *,
    charge: str,
    style: RenderStyle,
    scale: int,
    charge_font: ImageFont.FreeTypeFont,
) -> None:
    left = min(position[0] for position in positions) - 30 * scale
    right = max(position[0] for position in positions) + 30 * scale
    top = min(position[1] for position in positions) - 34 * scale
    bottom = max(position[1] for position in positions) + 34 * scale
    hook = 10 * scale
    width = max(1, round(style.line_width * scale))

    draw.line(((left + hook, top), (left, top), (left, bottom), (left + hook, bottom)),
              fill=style.bond_color, width=width, joint="curve")
    draw.line(((right - hook, top), (right, top), (right, bottom), (right - hook, bottom)),
              fill=style.bond_color, width=width, joint="curve")
    draw.text(
        (right + 6 * scale, top - 2 * scale),
        charge,
        font=charge_font,
        fill=style.atom_color,
        anchor="ls",
    )


def _add_paper_texture(
    image: Image.Image,
    *,
    speck_count: int,
    background: tuple[int, int, int],
    rng: random.Random,
) -> None:
    if speck_count <= 0:
        return

    draw = ImageDraw.Draw(image)
    for _ in range(speck_count):
        x = rng.randrange(image.width)
        y = rng.randrange(image.height)
        shade_delta = rng.choice((-1, 1)) * rng.randint(2, 7)
        color = tuple(
            min(255, max(0, channel + shade_delta)) for channel in background
        )
        radius = rng.choice((1, 1, 2))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def render_molecule(
    molecule: Molecule,
    *,
    size: int = DEFAULT_SIZE,
    style: RenderStyle,
    seed: int,
) -> Image.Image:
    """Render one molecule as a square RGB image."""
    if size < 64:
        raise ValueError("size must be at least 64 pixels")

    rng = random.Random(seed)
    scale = SUPERSAMPLE
    canvas_size = size * scale
    image = Image.new("RGB", (canvas_size, canvas_size), style.background)
    _add_paper_texture(
        image,
        speck_count=style.paper_specks * scale,
        background=style.background,
        rng=rng,
    )
    draw = ImageDraw.Draw(image)

    font_range = (22.0, 25.0) if len(molecule.atoms) >= 8 else (25.0, 28.0)
    font_size = round(rng.uniform(*font_range) * scale)
    atom_font = _load_font(font_size, style.font_names)
    charge_font = _load_font(round(17 * scale), style.font_names)
    maximum_x = max(abs(atom.x) for atom in molecule.atoms) + 0.45
    maximum_y = max(abs(atom.y) for atom in molecule.atoms) + 0.45
    maximum_extent = max(maximum_x, maximum_y, 1.0)
    logical_spacing = min(
        rng.uniform(49.0, 54.0),
        (size / 2 - 27) / maximum_extent,
    )
    spacing = logical_spacing * scale
    center_x = canvas_size / 2 + rng.uniform(-3.0, 3.0) * scale
    center_y = canvas_size / 2 + rng.uniform(-3.0, 3.0) * scale

    positions = [
        (center_x + atom.x * spacing, center_y + atom.y * spacing)
        for atom in molecule.atoms
    ]

    for bond in molecule.bonds:
        _draw_bond(
            draw,
            positions[bond.start],
            positions[bond.end],
            order=bond.order,
            style=style,
            scale=scale,
            rng=rng,
        )

    for atom, position in zip(molecule.atoms, positions):
        draw.text(
            position,
            atom.symbol,
            font=atom_font,
            fill=style.atom_color,
            anchor="mm",
        )
        if atom.formal_charge is not None:
            draw.text(
                (position[0] + 12 * scale, position[1] - 12 * scale),
                atom.formal_charge,
                font=charge_font,
                fill=style.atom_color,
                anchor="mm",
            )
        for angle in atom.lone_pair_angles:
            _draw_lone_pair(
                draw,
                position,
                angle,
                color=style.electron_color,
                scale=scale,
            )

    if molecule.overall_charge is not None:
        _draw_ion_brackets(
            draw,
            positions,
            charge=molecule.overall_charge,
            style=style,
            scale=scale,
            charge_font=charge_font,
        )

    rotation = rng.uniform(-style.rotation, style.rotation)
    if rotation:
        image = image.rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=style.background,
        )
    if style.blur_radius > 0:
        image = image.filter(
            ImageFilter.GaussianBlur(radius=style.blur_radius * scale)
        )

    return image.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")


def generate_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    count: int = DEFAULT_COUNT,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    """Generate one image per distinct structure and return their paths."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    molecules = _molecules()
    if not 1 <= count <= len(molecules):
        raise ValueError(
            f"count must be between 1 and {len(molecules)} unique structures"
        )

    for old_path in destination.glob("synthetic-lewis-*.png"):
        old_path.unlink()

    master_rng = random.Random(seed)
    generated_paths: list[Path] = []

    for index, molecule in enumerate(molecules[:count], start=1):
        style = master_rng.choice(STYLES)
        image_seed = master_rng.randrange(0, 2**32)
        image = render_molecule(
            molecule,
            size=size,
            style=style,
            seed=image_seed,
        )
        filename = (
            f"synthetic-lewis-{index:03d}-"
            f"{molecule.slug}-{style.name}.png"
        )
        output_path = destination / filename
        image.save(output_path, format="PNG", optimize=True)
        generated_paths.append(output_path)
        print(f"[{index}/{count}] Saved {output_path.name}")

    return generated_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Lewis-structure training PNG files."
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
        help=f"number of unique structures, at most 130 (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"image width and height (default: {DEFAULT_SIZE})",
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
        f"Generated {len(paths)} Lewis-structure images in "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
