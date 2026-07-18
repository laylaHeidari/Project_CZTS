#!/usr/bin/env python3
"""
Generate representative CZTS bond distortions for finite-difference TDDFPT.

The script:
1. reads a reference CZTS XYZ structure;
2. identifies Cu-S, Sn-S, and Zn-S bonds using periodic minimum-image distances;
3. selects representative bonds across each bond-length range;
4. creates positive and negative displaced structures and CP2K inputs; and
5. writes a table describing the selected coordinates.

Run:
    python generate_bond_displacements.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_CELL = (10.8683996201, 10.8683996201, 10.8495998383)
DEFAULT_BOND_SPECS = (
    ("Cu", "S", 2.70),
    ("Sn", "S", 3.00),
    ("Zn", "S", 2.80),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create plus/minus displaced CZTS structures and CP2K TDDFPT "
            "inputs for representative local bond-stretch coordinates."
        )
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path("czts_reference_structure.xyz"),
        help="Reference XYZ structure.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("cp2k_tddfpt_template.inp"),
        help="CP2K template containing {PROJECT} and {COORD_FILE}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calculations"),
        help="Directory for generated calculations.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Displacement amplitude in Angstrom along the normalized coordinate.",
    )
    parser.add_argument(
        "--n-per-type",
        type=int,
        default=3,
        help="Number of representative bonds selected for each bond type.",
    )
    parser.add_argument(
        "--cell",
        type=float,
        nargs=3,
        metavar=("A", "B", "C"),
        default=DEFAULT_CELL,
        help="Orthorhombic cell lengths in Angstrom.",
    )
    parser.add_argument(
        "--cu-s-cutoff",
        type=float,
        default=2.70,
        help="Cu-S bond cutoff in Angstrom.",
    )
    parser.add_argument(
        "--sn-s-cutoff",
        type=float,
        default=3.00,
        help="Sn-S bond cutoff in Angstrom.",
    )
    parser.add_argument(
        "--zn-s-cutoff",
        type=float,
        default=2.80,
        help="Zn-S bond cutoff in Angstrom.",
    )
    return parser.parse_args()


def read_xyz(path: Path) -> tuple[list[str], np.ndarray, str]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ file: {path}")

    atom_count = int(lines[0].strip())
    if len(lines) < atom_count + 2:
        raise ValueError(
            f"{path} declares {atom_count} atoms but contains too few coordinate lines."
        )

    elements: list[str] = []
    coordinates: list[list[float]] = []

    for line in lines[2 : atom_count + 2]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ coordinate line in {path}: {line}")
        elements.append(fields[0])
        coordinates.append([float(fields[1]), float(fields[2]), float(fields[3])])

    return elements, np.asarray(coordinates, dtype=float), lines[1]


def write_xyz(
    path: Path,
    elements: Iterable[str],
    coordinates: np.ndarray,
    comment: str,
) -> None:
    elements = list(elements)
    with path.open("w") as handle:
        handle.write(f"{len(elements)}\n")
        handle.write(f"{comment}\n")
        for element, xyz in zip(elements, coordinates):
            handle.write(
                f"{element:2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}\n"
            )


def minimum_image_vector(
    first: np.ndarray,
    second: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    vector = second - first
    vector -= cell * np.round(vector / cell)
    return vector


def find_bonds(
    elements: list[str],
    coordinates: np.ndarray,
    first_element: str,
    second_element: str,
    cutoff: float,
    cell: np.ndarray,
) -> list[tuple[int, int, str, str, float]]:
    pairs: list[tuple[int, int, str, str, float]] = []

    for first_index in range(len(elements)):
        for second_index in range(first_index + 1, len(elements)):
            pair_elements = sorted(
                [elements[first_index], elements[second_index]]
            )
            if pair_elements != sorted([first_element, second_element]):
                continue

            vector = minimum_image_vector(
                coordinates[first_index],
                coordinates[second_index],
                cell,
            )
            distance = float(np.linalg.norm(vector))

            if distance <= cutoff:
                pairs.append(
                    (
                        first_index + 1,
                        second_index + 1,
                        elements[first_index],
                        elements[second_index],
                        distance,
                    )
                )

    return sorted(pairs, key=lambda item: item[4])


def select_representative_bonds(
    pairs: list[tuple[int, int, str, str, float]],
    number_to_select: int,
) -> list[tuple[int, int, str, str, float]]:
    if number_to_select < 1:
        raise ValueError("--n-per-type must be at least 1.")

    if len(pairs) <= number_to_select:
        return pairs

    indices = np.linspace(
        0,
        len(pairs) - 1,
        number_to_select,
        dtype=int,
    )
    return [pairs[index] for index in dict.fromkeys(indices)]


def coordinate_label(
    element_i: str,
    element_j: str,
    atom_i: int,
    atom_j: int,
) -> str:
    return f"{element_i}{element_j}_{atom_i}_{atom_j}"


def normalized_bond_stretch(
    coordinates: np.ndarray,
    atom_i: int,
    atom_j: int,
    cell: np.ndarray,
) -> np.ndarray:
    first_index = atom_i - 1
    second_index = atom_j - 1

    vector = minimum_image_vector(
        coordinates[first_index],
        coordinates[second_index],
        cell,
    )
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError(f"Atoms {atom_i} and {atom_j} occupy the same position.")

    unit_vector = vector / norm
    mode = np.zeros_like(coordinates)
    mode[first_index] = -unit_vector / np.sqrt(2.0)
    mode[second_index] = unit_vector / np.sqrt(2.0)
    return mode


def render_cp2k_input(
    template_text: str,
    project_name: str,
    coordinate_filename: str,
) -> str:
    required_tokens = ("{PROJECT}", "{COORD_FILE}")
    missing = [token for token in required_tokens if token not in template_text]
    if missing:
        raise ValueError(
            "The CP2K template is missing placeholders: "
            + ", ".join(missing)
        )

    return (
        template_text.replace("{PROJECT}", project_name)
        .replace("{COORD_FILE}", coordinate_filename)
    )


def create_coordinate_calculations(
    output_dir: Path,
    label: str,
    elements: list[str],
    coordinates: np.ndarray,
    mode: np.ndarray,
    delta: float,
    template_text: str,
) -> None:
    coordinate_dir = output_dir / label

    for sign, displacement_name in ((1.0, "plus"), (-1.0, "minus")):
        displacement_dir = coordinate_dir / displacement_name
        displacement_dir.mkdir(parents=True, exist_ok=True)

        displaced_coordinates = coordinates + sign * delta * mode
        coordinate_filename = "structure.xyz"
        input_filename = "cp2k.inp"
        project_name = f"CZTS_FD_{label}_{displacement_name}"

        write_xyz(
            displacement_dir / coordinate_filename,
            elements,
            displaced_coordinates,
            (
                f"{displacement_name} displacement; "
                f"delta={delta:.6f} Angstrom; coordinate={label}"
            ),
        )

        cp2k_input = render_cp2k_input(
            template_text,
            project_name,
            coordinate_filename,
        )
        (displacement_dir / input_filename).write_text(cp2k_input)


def main() -> None:
    args = parse_arguments()

    if args.delta <= 0.0:
        raise ValueError("--delta must be greater than zero.")

    elements, coordinates, _ = read_xyz(args.structure)
    template_text = args.template.read_text()
    cell = np.asarray(args.cell, dtype=float)

    bond_specs = (
        ("Cu", "S", args.cu_s_cutoff),
        ("Sn", "S", args.sn_s_cutoff),
        ("Zn", "S", args.zn_s_cutoff),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict] = []

    for first_element, second_element, cutoff in bond_specs:
        bonds = find_bonds(
            elements,
            coordinates,
            first_element,
            second_element,
            cutoff,
            cell,
        )
        selected_bonds = select_representative_bonds(
            bonds,
            args.n_per_type,
        )

        print(
            f"{first_element}-{second_element}: "
            f"found {len(bonds)}, selected {len(selected_bonds)}"
        )

        for atom_i, atom_j, element_i, element_j, distance in selected_bonds:
            label = coordinate_label(
                element_i,
                element_j,
                atom_i,
                atom_j,
            )
            mode = normalized_bond_stretch(
                coordinates,
                atom_i,
                atom_j,
                cell,
            )
            create_coordinate_calculations(
                output_dir=args.output_dir,
                label=label,
                elements=elements,
                coordinates=coordinates,
                mode=mode,
                delta=args.delta,
                template_text=template_text,
            )

            selected_rows.append(
                {
                    "label": label,
                    "element_i": element_i,
                    "element_j": element_j,
                    "atom_i": atom_i,
                    "atom_j": atom_j,
                    "distance_A": f"{distance:.8f}",
                }
            )

    coordinates_csv = args.output_dir / "selected_coordinates.csv"
    with coordinates_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "element_i",
                "element_j",
                "atom_i",
                "atom_j",
                "distance_A",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"Wrote: {coordinates_csv}")
    print(
        "Each plus/minus directory contains structure.xyz and cp2k.inp. "
        "Run CP2K in each directory and save the output as out_cp2k.log."
    )


if __name__ == "__main__":
    main()
