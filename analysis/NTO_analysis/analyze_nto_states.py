#!/usr/bin/env python3
"""
Analyze hole and particle NTO cube files for excited states detected
from the cube-file names.

For an excited state represented by more than one NTO pair, enter the
corresponding NTO-pair weights in NTO_WEIGHTS below. These weights are
obtained from the NTO calculation output and are normalized internally.

Run:
    python analyze_nto_states.py

Outputs:
    CZTS_NTO_states_summary.csv
    CZTS_NTO_weights_used.csv
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

# Use Path(".") when the cube files are in the directory from which the
# script is run. Replace it with another directory when needed, for example:
# INPUT_DIRECTORY = Path("/path/to/NTO/cube/files")
INPUT_DIRECTORY = Path(".")

# The output CSV files are written here.
OUTPUT_DIRECTORY = Path(".")

CUBE_PATTERN = "*.cube"
OUTPUT_PREFIX = "CZTS_NTO"

# Use periodic minimum-image distances when assigning grid density
# to the nearest atom.
USE_PBC = True

# Number of cube-grid points processed at one time.
CHUNK_SIZE = 20000

# NTO-pair weights obtained from the NTO calculation output.
# Entries are needed only for excited states represented by more than
# one NTO pair. The script normalizes the weights internally.
NTO_WEIGHTS = {
    "11801": 0.72765,
    "11802": 0.18106,
    "11803": 0.04653,
}


Z_TO_ELEM = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    57: "La",
    58: "Ce",
}


def read_cube(filename):
    """Read atoms, grid information, and NTO amplitudes from a cube file."""

    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as cube_file:
        # The first two cube-file lines are comments.
        cube_file.readline()
        cube_file.readline()

        parts = cube_file.readline().split()
        if len(parts) < 4:
            raise ValueError(
                f"{filename}: invalid atom-count/origin line."
            )

        natoms = int(parts[0])
        origin = np.array(
            [float(parts[1]), float(parts[2]), float(parts[3])],
            dtype=float,
        )

        grid_counts = []
        axes = []

        for _ in range(3):
            parts = cube_file.readline().split()
            if len(parts) < 4:
                raise ValueError(
                    f"{filename}: invalid cube-grid definition."
                )

            grid_counts.append(abs(int(parts[0])))
            axes.append(
                [float(parts[1]), float(parts[2]), float(parts[3])]
            )

        axes = np.asarray(axes, dtype=float)
        nx, ny, nz = grid_counts

        atoms = []

        for _ in range(abs(natoms)):
            parts = cube_file.readline().split()
            if len(parts) < 5:
                raise ValueError(f"{filename}: invalid atom line.")

            atomic_number = int(float(parts[0]))
            coordinate = np.array(
                [float(parts[2]), float(parts[3]), float(parts[4])],
                dtype=float,
            )

            atoms.append(
                {
                    "Z": atomic_number,
                    "element": Z_TO_ELEM.get(
                        atomic_number,
                        f"Z{atomic_number}",
                    ),
                    "coord": coordinate,
                }
            )

        values = []

        for line in cube_file:
            values.extend(float(value) for value in line.split())

    expected_values = nx * ny * nz

    if len(values) != expected_values:
        raise ValueError(
            f"{filename}: expected {expected_values} grid values, "
            f"but found {len(values)}."
        )

    data = np.asarray(values, dtype=float).reshape((nx, ny, nz))

    return atoms, origin, axes, data


def find_nto_pairs(input_directory, cube_pattern):
    """
    Detect complete hole/particle NTO pairs from cube-file names.

    The expected identifying part of a file name is:

        _00101_Hole_State
        _00101_Particle_State

    The first three digits identify the excited state and the following
    two digits identify the NTO pair.
    """

    cube_files = sorted(input_directory.glob(cube_pattern))

    if not cube_files:
        raise FileNotFoundError(
            f"No cube files matching '{cube_pattern}' were found in "
            f"{input_directory.resolve()}."
        )

    filename_regex = re.compile(
        r"_(\d{3})(\d{2})_(Hole|Particle)_State",
        re.IGNORECASE,
    )

    hole_files = {}
    particle_files = {}

    for filename in cube_files:
        match = filename_regex.search(filename.name)

        if not match:
            continue

        state_number = int(match.group(1))
        pair_number = match.group(2)
        cube_type = match.group(3).lower()

        pair_id = f"{state_number:03d}{pair_number}"
        key = (state_number, pair_id)

        target = hole_files if cube_type == "hole" else particle_files

        if key in target:
            raise ValueError(
                f"More than one {cube_type} cube was detected for "
                f"NTO pair {pair_id}."
            )

        target[key] = filename

    complete_keys = sorted(set(hole_files) & set(particle_files))

    if not complete_keys:
        raise FileNotFoundError(
            "No complete hole/particle NTO cube pairs were detected. "
            "Check the cube-file names."
        )

    incomplete_holes = sorted(set(hole_files) - set(particle_files))
    incomplete_particles = sorted(set(particle_files) - set(hole_files))

    if incomplete_holes or incomplete_particles:
        messages = []

        if incomplete_holes:
            messages.append(
                "missing particle cubes for "
                + ", ".join(pair_id for _, pair_id in incomplete_holes)
            )

        if incomplete_particles:
            messages.append(
                "missing hole cubes for "
                + ", ".join(pair_id for _, pair_id in incomplete_particles)
            )

        raise FileNotFoundError(
            "Incomplete NTO pairs were detected: " + "; ".join(messages)
        )

    pairs_by_state = {}

    for state_number, pair_id in complete_keys:
        pairs_by_state.setdefault(state_number, []).append(
            (
                pair_id,
                hole_files[(state_number, pair_id)],
                particle_files[(state_number, pair_id)],
            )
        )

    return pairs_by_state


def get_normalized_weights(pairs, nto_weights):
    """
    Return raw and normalized weights for one excited state.

    A state represented by one NTO pair automatically receives weight 1.
    A state represented by multiple pairs must have a weight for every pair
    in NTO_WEIGHTS.
    """

    pair_ids = [pair_id for pair_id, _, _ in pairs]

    if len(pair_ids) == 1:
        raw_weights = np.array([1.0], dtype=float)
        weight_note = "single_pair_weight_one"
    else:
        missing = [
            pair_id for pair_id in pair_ids if pair_id not in nto_weights
        ]

        if missing:
            raise ValueError(
                "Missing NTO-pair weights for: "
                + ", ".join(missing)
                + ". Add them to NTO_WEIGHTS."
            )

        raw_weights = np.array(
            [nto_weights[pair_id] for pair_id in pair_ids],
            dtype=float,
        )
        weight_note = "nto_weights_used"

    if np.any(raw_weights < 0):
        raise ValueError("NTO-pair weights must be nonnegative.")

    weight_sum = raw_weights.sum()

    if weight_sum <= 0:
        raise ValueError(
            "The sum of the NTO-pair weights must be greater than zero."
        )

    normalized_weights = raw_weights / weight_sum

    return raw_weights, normalized_weights, weight_note


def validate_cube_compatibility(
    filename,
    atoms,
    origin,
    axes,
    data,
    reference_atoms,
    reference_origin,
    reference_axes,
    reference_shape,
):
    """Confirm that cube files used in one weighted density are compatible."""

    if data.shape != reference_shape:
        raise ValueError(f"Cube-grid shape mismatch in {filename}.")

    if len(atoms) != len(reference_atoms):
        raise ValueError(f"Atom-count mismatch in {filename}.")

    if not np.allclose(origin, reference_origin):
        raise ValueError(f"Cube-origin mismatch in {filename}.")

    if not np.allclose(axes, reference_axes):
        raise ValueError(f"Cube-axis mismatch in {filename}.")

    atomic_numbers = [atom["Z"] for atom in atoms]
    reference_atomic_numbers = [atom["Z"] for atom in reference_atoms]

    if atomic_numbers != reference_atomic_numbers:
        raise ValueError(f"Atomic-number mismatch in {filename}.")

    coordinates = np.array([atom["coord"] for atom in atoms], dtype=float)
    reference_coordinates = np.array(
        [atom["coord"] for atom in reference_atoms],
        dtype=float,
    )

    if not np.allclose(coordinates, reference_coordinates):
        raise ValueError(f"Atomic-coordinate mismatch in {filename}.")


def build_weighted_density(
    pairs,
    raw_weights,
    normalized_weights,
    density_type,
):
    """
    Construct the weighted hole or particle NTO density:

        rho_total(r) = sum_k lambda_k |psi_k(r)|^2
    """

    reference_atoms = None
    reference_origin = None
    reference_axes = None
    reference_shape = None
    total_density = None
    pair_information = []

    for index, (pair_id, hole_file, particle_file) in enumerate(pairs):
        if density_type == "hole":
            cube_filename = hole_file
        elif density_type == "particle":
            cube_filename = particle_file
        else:
            raise ValueError(
                "density_type must be either 'hole' or 'particle'."
            )

        atoms, origin, axes, cube_data = read_cube(cube_filename)

        if reference_atoms is None:
            reference_atoms = atoms
            reference_origin = origin
            reference_axes = axes
            reference_shape = cube_data.shape
            total_density = np.zeros_like(cube_data, dtype=float)
        else:
            validate_cube_compatibility(
                filename=cube_filename,
                atoms=atoms,
                origin=origin,
                axes=axes,
                data=cube_data,
                reference_atoms=reference_atoms,
                reference_origin=reference_origin,
                reference_axes=reference_axes,
                reference_shape=reference_shape,
            )

        # The cube contains an NTO amplitude psi_k(r). Convert it to
        # density and add its normalized weighted contribution.
        pair_density = cube_data**2
        total_density += normalized_weights[index] * pair_density

        pair_information.append(
            {
                "pair_id": pair_id,
                "raw_weight": raw_weights[index],
                "normalized_weight_lambda": normalized_weights[index],
            }
        )

    return (
        reference_atoms,
        reference_origin,
        reference_axes,
        total_density,
        pair_information,
    )


def assign_density_to_atoms(
    atoms,
    origin,
    axes,
    density,
    pbc,
    chunk_size,
):
    """
    Assign integrated NTO density to the nearest atoms and calculate
    element percentages and the atom-based participation ratio.
    """

    atom_coordinates = np.array(
        [atom["coord"] for atom in atoms],
        dtype=float,
    )
    elements = [atom["element"] for atom in atoms]
    number_of_atoms = len(atoms)

    nx, ny, nz = density.shape

    cell = np.vstack(
        [
            axes[0] * nx,
            axes[1] * ny,
            axes[2] * nz,
        ]
    )

    inverse_cell = np.linalg.inv(cell)
    voxel_volume = abs(np.linalg.det(axes))

    atom_density_sums = np.zeros(number_of_atoms, dtype=float)
    flat_density = density.ravel()
    total_grid_points = nx * ny * nz

    for start in range(0, total_grid_points, chunk_size):
        end = min(start + chunk_size, total_grid_points)
        indices = np.arange(start, end)

        ix = indices // (ny * nz)
        remainder = indices % (ny * nz)
        iy = remainder // nz
        iz = remainder % nz

        grid_points = (
            origin
            + ix[:, None] * axes[0]
            + iy[:, None] * axes[1]
            + iz[:, None] * axes[2]
        )

        if pbc:
            grid_fractional = grid_points @ inverse_cell
            atoms_fractional = atom_coordinates @ inverse_cell

            fractional_difference = (
                grid_fractional[:, None, :]
                - atoms_fractional[None, :, :]
            )
            fractional_difference -= np.round(fractional_difference)

            cartesian_difference = fractional_difference @ cell
            squared_distances = np.sum(
                cartesian_difference**2,
                axis=2,
            )
        else:
            coordinate_difference = (
                grid_points[:, None, :]
                - atom_coordinates[None, :, :]
            )
            squared_distances = np.sum(
                coordinate_difference**2,
                axis=2,
            )

        nearest_atoms = np.argmin(squared_distances, axis=1)
        integrated_grid_density = (
            flat_density[start:end] * voxel_volume
        )

        np.add.at(
            atom_density_sums,
            nearest_atoms,
            integrated_grid_density,
        )

    total_integrated_density = atom_density_sums.sum()

    if total_integrated_density <= 0:
        raise ValueError(
            "The total integrated density is zero or negative."
        )

    atom_weights = atom_density_sums / total_integrated_density

    atom_dataframe = pd.DataFrame(
        {
            "atom_index": np.arange(1, number_of_atoms + 1),
            "element": elements,
            "integrated_density_q_i": atom_density_sums,
            "atom_weight_w_i": atom_weights,
            "percent": 100.0 * atom_weights,
        }
    )

    element_dataframe = (
        atom_dataframe.groupby("element", as_index=False)
        .agg(
            integrated_density_Q_E=(
                "integrated_density_q_i",
                "sum",
            )
        )
    )

    element_dataframe["percent"] = (
        100.0
        * element_dataframe["integrated_density_Q_E"]
        / total_integrated_density
    )
    element_dataframe = element_dataframe.sort_values("element")

    participation_ratio = 1.0 / np.sum(atom_weights**2)

    return (
        element_dataframe,
        participation_ratio,
        total_integrated_density,
    )


def get_element_percent(element_dataframe, element):
    """Return the percentage assigned to an element."""

    result = element_dataframe.loc[
        element_dataframe["element"] == element,
        "percent",
    ]

    return 0.0 if result.empty else float(result.iloc[0])


def analyze_state(state, pairs):
    """Analyze the complete NTO pairs detected for one excited state."""

    pair_ids = [pair_id for pair_id, _, _ in pairs]

    raw_weights, normalized_weights, weight_note = (
        get_normalized_weights(pairs, NTO_WEIGHTS)
    )

    (
        atoms,
        origin,
        axes,
        hole_density,
        hole_pair_information,
    ) = build_weighted_density(
        pairs=pairs,
        raw_weights=raw_weights,
        normalized_weights=normalized_weights,
        density_type="hole",
    )

    (
        particle_atoms,
        particle_origin,
        particle_axes,
        particle_density,
        _,
    ) = build_weighted_density(
        pairs=pairs,
        raw_weights=raw_weights,
        normalized_weights=normalized_weights,
        density_type="particle",
    )

    # Confirm that the total hole and particle densities use the same
    # atomic structure and cube grid.
    validate_cube_compatibility(
        filename=f"particle cubes for S{state}",
        atoms=particle_atoms,
        origin=particle_origin,
        axes=particle_axes,
        data=particle_density,
        reference_atoms=atoms,
        reference_origin=origin,
        reference_axes=axes,
        reference_shape=hole_density.shape,
    )

    (
        hole_element_dataframe,
        hole_participation_ratio,
        hole_total_integral,
    ) = assign_density_to_atoms(
        atoms=atoms,
        origin=origin,
        axes=axes,
        density=hole_density,
        pbc=USE_PBC,
        chunk_size=CHUNK_SIZE,
    )

    (
        particle_element_dataframe,
        particle_participation_ratio,
        particle_total_integral,
    ) = assign_density_to_atoms(
        atoms=atoms,
        origin=origin,
        axes=axes,
        density=particle_density,
        pbc=USE_PBC,
        chunk_size=CHUNK_SIZE,
    )

    elements = sorted(
        set(hole_element_dataframe["element"])
        | set(particle_element_dataframe["element"])
    )

    summary_row = {
        "state": f"S{state}",
        "state_number": state,
        "nto_pairs_included": ";".join(pair_ids),
        "number_of_pairs": len(pair_ids),
        "weight_note": weight_note,
        "hole_PR": hole_participation_ratio,
        "particle_PR": particle_participation_ratio,
        "hole_total_integral": hole_total_integral,
        "particle_total_integral": particle_total_integral,
    }

    for element in elements:
        summary_row[f"{element}_hole_percent"] = get_element_percent(
            hole_element_dataframe,
            element,
        )
        summary_row[f"{element}_particle_percent"] = get_element_percent(
            particle_element_dataframe,
            element,
        )

    summary_row["Cu_Zn_hole_percent"] = (
        summary_row.get("Cu_hole_percent", 0.0)
        + summary_row.get("Zn_hole_percent", 0.0)
    )

    summary_row["Cu_Zn_S_hole_percent"] = (
        summary_row.get("Cu_hole_percent", 0.0)
        + summary_row.get("Zn_hole_percent", 0.0)
        + summary_row.get("S_hole_percent", 0.0)
    )

    summary_row["Sn_S_particle_percent"] = (
        summary_row.get("Sn_particle_percent", 0.0)
        + summary_row.get("S_particle_percent", 0.0)
    )

    weight_rows = []

    for pair_information in hole_pair_information:
        weight_rows.append(
            {
                "state": f"S{state}",
                "state_number": state,
                "pair_id": pair_information["pair_id"],
                "raw_weight": pair_information["raw_weight"],
                "normalized_weight_lambda": pair_information[
                    "normalized_weight_lambda"
                ],
                "weight_note": weight_note,
            }
        )

    return summary_row, weight_rows


def main():
    """Run the analysis for the excited states detected in the cube files."""

    input_directory = INPUT_DIRECTORY.expanduser()
    output_directory = OUTPUT_DIRECTORY.expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    pairs_by_state = find_nto_pairs(
        input_directory=input_directory,
        cube_pattern=CUBE_PATTERN,
    )

    states = sorted(pairs_by_state)

    print("Excited states detected from the cube files:")
    print(" ".join(f"S{state}" for state in states))

    all_summary_rows = []
    all_weight_rows = []

    for state in states:
        print(f"\nAnalyzing S{state} ...")

        summary_row, weight_rows = analyze_state(
            state=state,
            pairs=pairs_by_state[state],
        )

        all_summary_rows.append(summary_row)
        all_weight_rows.extend(weight_rows)

        print(
            f"  NTO pairs: {summary_row['nto_pairs_included']}"
        )
        print(
            "  S hole percentage: "
            f"{summary_row.get('S_hole_percent', 0.0):.2f}"
        )
        print(
            "  Cu + Zn hole percentage: "
            f"{summary_row.get('Cu_Zn_hole_percent', 0.0):.2f}"
        )
        print(
            "  Sn + S particle percentage: "
            f"{summary_row.get('Sn_S_particle_percent', 0.0):.2f}"
        )
        print(
            f"  Hole PR: {summary_row['hole_PR']:.2f}; "
            f"Particle PR: {summary_row['particle_PR']:.2f}"
        )

    summary_dataframe = pd.DataFrame(
        all_summary_rows
    ).sort_values("state_number")

    weights_dataframe = pd.DataFrame(
        all_weight_rows
    ).sort_values(["state_number", "pair_id"])

    summary_filename = (
        output_directory / f"{OUTPUT_PREFIX}_states_summary.csv"
    )
    weights_filename = (
        output_directory / f"{OUTPUT_PREFIX}_weights_used.csv"
    )

    summary_dataframe.to_csv(summary_filename, index=False)
    weights_dataframe.to_csv(weights_filename, index=False)

    print("\nAnalysis completed.")
    print(f"Wrote: {summary_filename}")
    print(f"Wrote: {weights_filename}")


if __name__ == "__main__":
    main()
