#!/usr/bin/env python3
"""
Analyze finite-difference projected excited-state force corrections.

For each selected structural coordinate, the script reads one CP2K TDDFPT
output from a positive displacement and one from a negative displacement.
It then evaluates

    Delta F_I,Q = -[Omega_I(Q + Delta Q) - Omega_I(Q - Delta Q)] / (2 Delta Q)

for the requested excited states.

Default usage:
    python analyze_projected_force_corrections.py

Custom usage:
    python analyze_projected_force_corrections.py \
        --calculations-dir calculations \
        --coordinates-file selected_coordinates.csv \
        --delta 0.01 \
        --states 1 2 3 4 80 118 \
        --output-dir results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List


DEFAULT_STATES = [1, 2, 3, 4, 80, 118]
DEFAULT_DELTA_ANGSTROM = 0.01
DEFAULT_LOG_PATTERN = "out_*.log"

ALL_RESULTS_NAME = "projected_force_results_all_coordinates.csv"
SUMMARY_NAME = "projected_force_summary_by_state.csv"

REQUIRED_COORDINATE_COLUMNS = {
    "label",
    "element_i",
    "element_j",
    "atom_i",
    "atom_j",
    "distance_A",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate projected excited-state force corrections from "
            "plus/minus CP2K TDDFPT calculations."
        )
    )
    parser.add_argument(
        "--calculations-dir",
        type=Path,
        default=Path("calculations"),
        help=(
            "Directory containing one subdirectory per coordinate. "
            "Each coordinate directory must contain plus/ and minus/ folders."
        ),
    )
    parser.add_argument(
        "--coordinates-file",
        type=Path,
        default=Path("selected_coordinates.csv"),
        help="CSV file describing the selected structural coordinates.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA_ANGSTROM,
        help="Magnitude of each positive/negative displacement in Angstrom.",
    )
    parser.add_argument(
        "--states",
        type=int,
        nargs="+",
        default=DEFAULT_STATES,
        help="One-based TDDFPT excited-state indices to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory in which the output CSV files will be written.",
    )
    parser.add_argument(
        "--log-pattern",
        default=DEFAULT_LOG_PATTERN,
        help="Filename pattern used to locate CP2K output logs.",
    )
    return parser.parse_args()


def extract_last_tddfpt_table(logfile: Path) -> Dict[int, float]:
    """Extract the last TDDFPT excitation-energy table from a CP2K output."""
    lines = logfile.read_text(errors="ignore").splitlines()

    last_table: Dict[int, float] = {}
    current_table: Dict[int, float] = {}
    inside_table = False

    for line in lines:
        if (
            "State" in line
            and "Exc. energy (eV)" in line
            and "Convergence" in line
        ):
            if current_table:
                last_table = current_table
            current_table = {}
            inside_table = True
            continue

        if not inside_table:
            continue

        parts = line.split()
        if len(parts) < 3:
            if current_table:
                last_table = current_table
            current_table = {}
            inside_table = False
            continue

        try:
            state = int(parts[0])
            excitation_energy_ev = float(parts[1])
            float(parts[2])
        except ValueError:
            if current_table:
                last_table = current_table
            current_table = {}
            inside_table = False
            continue

        current_table[state] = excitation_energy_ev

    if current_table:
        last_table = current_table

    if not last_table:
        raise RuntimeError(
            f"No TDDFPT excitation-energy table was found in: {logfile}"
        )

    return last_table


def find_single_log(folder: Path, pattern: str) -> Path:
    logs = sorted(folder.glob(pattern))
    if len(logs) != 1:
        raise RuntimeError(
            f"Expected exactly one log matching '{pattern}' in {folder}, "
            f"but found {len(logs)}: {logs}"
        )
    return logs[0]


def read_coordinates_table(csv_path: Path) -> List[dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Coordinate table not found: {csv_path}")

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COORDINATE_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"No coordinate rows were found in: {csv_path}")

    return rows


def analyze_coordinates(
    coordinate_rows: Iterable[dict],
    calculations_dir: Path,
    states: Iterable[int],
    delta: float,
    log_pattern: str,
) -> List[dict]:
    if delta <= 0.0:
        raise ValueError("--delta must be greater than zero.")

    all_results: List[dict] = []

    for coordinate in coordinate_rows:
        label = coordinate["label"].strip()
        coordinate_dir = calculations_dir / label

        try:
            plus_log = find_single_log(coordinate_dir / "plus", log_pattern)
            minus_log = find_single_log(coordinate_dir / "minus", log_pattern)
            plus_energies = extract_last_tddfpt_table(plus_log)
            minus_energies = extract_last_tddfpt_table(minus_log)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Skipping coordinate '{label}': {exc}")
            continue

        for state in states:
            if state not in plus_energies or state not in minus_energies:
                print(
                    f"Skipping state {state} for coordinate '{label}': "
                    "state not present in both TDDFPT outputs."
                )
                continue

            omega_plus = plus_energies[state]
            omega_minus = minus_energies[state]
            derivative = (omega_plus - omega_minus) / (2.0 * delta)
            projected_force_correction = -derivative

            all_results.append(
                {
                    "state": state,
                    "coordinate_label": label,
                    "bond_type": (
                        f"{coordinate['element_i'].strip()}-"
                        f"{coordinate['element_j'].strip()}"
                    ),
                    "atom_i": coordinate["atom_i"],
                    "atom_j": coordinate["atom_j"],
                    "reference_distance_A": coordinate["distance_A"],
                    "displacement_A": f"{delta:.8f}",
                    "omega_minus_eV": f"{omega_minus:.8f}",
                    "omega_plus_eV": f"{omega_plus:.8f}",
                    "projected_force_correction_eV_per_A": (
                        f"{projected_force_correction:.8f}"
                    ),
                    "abs_projected_force_correction_eV_per_A": (
                        f"{abs(projected_force_correction):.8f}"
                    ),
                }
            )

    return all_results


def summarize_by_state(all_results: List[dict], states: Iterable[int]) -> List[dict]:
    summary_rows: List[dict] = []

    for state in states:
        state_rows = [
            row for row in all_results if int(row["state"]) == int(state)
        ]
        if not state_rows:
            continue

        values = [
            float(row["abs_projected_force_correction_eV_per_A"])
            for row in state_rows
        ]
        min_row = min(
            state_rows,
            key=lambda row: float(
                row["abs_projected_force_correction_eV_per_A"]
            ),
        )
        max_row = max(
            state_rows,
            key=lambda row: float(
                row["abs_projected_force_correction_eV_per_A"]
            ),
        )

        summary_rows.append(
            {
                "state": state,
                "n_coordinates": len(values),
                "min_abs_correction_eV_per_A": f"{min(values):.8f}",
                "mean_abs_correction_eV_per_A": f"{mean(values):.8f}",
                "median_abs_correction_eV_per_A": f"{median(values):.8f}",
                "max_abs_correction_eV_per_A": f"{max(values):.8f}",
                "min_coordinate": min_row["coordinate_label"],
                "max_coordinate": max_row["coordinate_label"],
                "max_bond_type": max_row["bond_type"],
            }
        )

    return summary_rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows are available for output: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_arguments()

    coordinate_rows = read_coordinates_table(args.coordinates_file)
    all_results = analyze_coordinates(
        coordinate_rows=coordinate_rows,
        calculations_dir=args.calculations_dir,
        states=args.states,
        delta=args.delta,
        log_pattern=args.log_pattern,
    )

    if not all_results:
        raise RuntimeError(
            "No projected-force results were produced. Check the coordinate "
            "labels, directory structure, log filenames, and requested states."
        )

    summary_rows = summarize_by_state(all_results, args.states)

    all_results_path = args.output_dir / ALL_RESULTS_NAME
    summary_path = args.output_dir / SUMMARY_NAME

    write_csv(all_results_path, all_results)
    write_csv(summary_path, summary_rows)

    print(f"Wrote: {all_results_path}")
    print(f"Wrote: {summary_path}")
    print(f"Analyzed coordinate-state pairs: {len(all_results)}")


if __name__ == "__main__":
    main()
