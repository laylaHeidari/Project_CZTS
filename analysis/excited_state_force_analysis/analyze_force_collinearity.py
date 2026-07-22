#!/usr/bin/env python3

"""
NBRA force-collinearity analysis for CZTS.

This script computes the force-vector collinearity between ground-state
and excited-state CP2K force calculations for several representative
MD snapshots.

For each snapshot and excited state i, the script reads the CP2K .forces
files for the ground state and excited state. The atomic force components
are concatenated into one 3N-dimensional vector:

    F = (F1x, F1y, F1z, F2x, F2y, F2z, ..., FNx, FNy, FNz)

where N is the number of atoms.

The force collinearity is calculated as:

    cos(theta_i) = (F0 . Fi) / (|F0| |Fi|)

where:
    F0 = ground-state force vector
    Fi = excited-state force vector for electronic state i

The script computes cos(theta_i) for each available snapshot/state pair,
then averages the cosine values over the selected geometries and computes
the sample variance:

    Var[cos(theta_i)] =
    sum_k (cos(theta_i,k) - <cos(theta_i)>)^2 / (n - 1)

where n is the number of geometries used for that electronic state.

The final SI table is written to:

    czts_nbra_force_collinearity_summary.csv

This table contains:
    state
    n_geometries
    average_cos_theta
    variance_cos_theta

The individual per-snapshot values are written to:

    czts_nbra_force_collinearity_by_snapshot.csv

This file contains:
    snapshot
    state
    cos_theta
    GS_force_file
    ES_force_file

Missing files or atom-order mismatches are written to:

    czts_nbra_force_collinearity_missing_files.csv

No force-unit conversion is needed for cos(theta), because the cosine
similarity is unitless and any constant force-unit conversion cancels
in the ratio.
"""

import argparse
import csv
import math
import re
from pathlib import Path


def parse_cp2k_forces(force_file):
    """
    Read a CP2K .forces file.

    Expected force rows:
        atom_index   kind_index   element   Fx   Fy   Fz

    Returns:
        atoms: list of (atom_index, kind_index, element)
        forces_flat: 3N-dimensional force vector
    """
    atoms = []
    forces_flat = []

    with open(force_file, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()

            if len(parts) == 6 and parts[0].isdigit():
                try:
                    atom_index = int(parts[0])
                    kind_index = int(parts[1])
                    element = parts[2]

                    fx = float(parts[3])
                    fy = float(parts[4])
                    fz = float(parts[5])

                    atoms.append((atom_index, kind_index, element))
                    forces_flat.extend([fx, fy, fz])

                except ValueError:
                    continue

    if len(forces_flat) == 0:
        raise ValueError(f"No force rows found in {force_file}")

    return atoms, forces_flat


def dot_product(v1, v2):
    """
    Dot product between two 3N-dimensional force vectors.
    """
    return sum(a * b for a, b in zip(v1, v2))


def norm(v):
    """
    Euclidean norm of a 3N-dimensional force vector.
    """
    return math.sqrt(sum(x * x for x in v))


def cos_theta(force_0, force_i):
    """
    Compute force collinearity:

        cos(theta_i) = (F0 . Fi) / (|F0| |Fi|)

    where:
        F0 = ground-state force vector
        Fi = excited-state force vector for electronic state i
    """
    n0 = norm(force_0)
    ni = norm(force_i)

    if n0 == 0.0 or ni == 0.0:
        return None

    return dot_product(force_0, force_i) / (n0 * ni)


def identify_state(path):
    """
    Identify GS, S1, S2, S3, S4, S80, S118, etc. from file/folder path.
    """
    text = str(path)

    if re.search(r"(^|[/_\-\s])GS([/_\-\s.]|$)", text, re.IGNORECASE):
        return "GS"

    if re.search(r"ground[_\-\s]*state", text, re.IGNORECASE):
        return "GS"

    matches = re.findall(r"(^|[/_\-\s])S(\d+)([/_\-\s.]|$)", text, re.IGNORECASE)

    if matches:
        return "S" + str(int(matches[-1][1]))

    return None


def identify_snapshot(path, root):
    """
    The snapshot name is assumed to be the first folder below the root.

    Example:
        root/snapshot_1182/S1/file.forces
        snapshot = snapshot_1182
    """
    path = Path(path).resolve()
    root = Path(root).resolve()

    rel_parts = path.relative_to(root).parts

    if len(rel_parts) < 2:
        return "unknown_snapshot"

    return rel_parts[0]


def collect_force_files(root):
    """
    Recursively collect all .forces files under root.

    Returns:
        data[snapshot][state] = force_file
    """
    root = Path(root).resolve()
    force_files = sorted(root.rglob("*.forces"))

    data = {}

    for force_file in force_files:
        snapshot = identify_snapshot(force_file, root)
        state = identify_state(force_file)

        if state is None:
            print(f"WARNING: Could not identify state for {force_file}")
            continue

        if snapshot not in data:
            data[snapshot] = {}

        if state in data[snapshot]:
            print("WARNING: Duplicate force file found")
            print(f"  snapshot: {snapshot}")
            print(f"  state:    {state}")
            print(f"  keeping:  {data[snapshot][state]}")
            print(f"  ignoring: {force_file}")
            continue

        data[snapshot][state] = force_file

    return data


def average(values):
    """
    Arithmetic average.
    """
    return sum(values) / len(values)


def sample_variance(values):
    """
    Sample variance over the selected geometries:

        Var = sum((x_i - mean)^2)/(n - 1)

    If only one geometry is available, variance is reported as 0.
    """
    n = len(values)

    if n <= 1:
        return 0.0

    avg = average(values)
    return sum((x - avg) ** 2 for x in values) / (n - 1)


def state_sort_key(state):
    """
    Sort states numerically:
        S1, S2, S3, S4, S80, S118
    """
    return int(state.replace("S", ""))


def main():
    parser = argparse.ArgumentParser(
        description="Compute cos(theta) between GS and excited-state CP2K force vectors."
    )

    parser.add_argument(
        "--root",
        default=".",
        help="Root folder containing snapshot folders. Default: current folder."
    )

    parser.add_argument(
        "--states",
        nargs="+",
        default=["S1", "S2", "S3", "S4", "S80", "S118"],
        help="Excited states to analyze."
    )

    parser.add_argument(
        "--out_prefix",
        default="czts_nbra_force_collinearity",
        help="Output file prefix."
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    states = args.states

    data = collect_force_files(root)

    by_snapshot_rows = []
    missing_rows = []

    for snapshot in sorted(data.keys()):
        files = data[snapshot]

        if "GS" not in files:
            missing_rows.append({
                "snapshot": snapshot,
                "state": "GS",
                "problem": "Missing GS force file"
            })
            continue

        try:
            atoms_gs, forces_gs = parse_cp2k_forces(files["GS"])
        except Exception as exc:
            missing_rows.append({
                "snapshot": snapshot,
                "state": "GS",
                "problem": f"Could not read GS force file: {exc}"
            })
            continue

        for state in states:
            if state not in files:
                missing_rows.append({
                    "snapshot": snapshot,
                    "state": state,
                    "problem": "Missing excited-state force file"
                })
                continue

            try:
                atoms_es, forces_es = parse_cp2k_forces(files[state])
            except Exception as exc:
                missing_rows.append({
                    "snapshot": snapshot,
                    "state": state,
                    "problem": f"Could not read ES force file: {exc}"
                })
                continue

            if atoms_es != atoms_gs:
                missing_rows.append({
                    "snapshot": snapshot,
                    "state": state,
                    "problem": "Atom order mismatch between GS and ES force files"
                })
                continue

            c = cos_theta(forces_gs, forces_es)

            if c is None:
                missing_rows.append({
                    "snapshot": snapshot,
                    "state": state,
                    "problem": "Zero force-vector norm"
                })
                continue

            by_snapshot_rows.append({
                "snapshot": snapshot,
                "state": state,
                "cos_theta": c,
                "GS_force_file": str(files["GS"]),
                "ES_force_file": str(files[state])
            })

    if len(by_snapshot_rows) == 0:
        raise RuntimeError("No valid GS/ES force-file pairs found.")

    # Write individual cos(theta) values for every snapshot/state pair.
    by_snapshot_file = args.out_prefix + "_by_snapshot.csv"

    with open(by_snapshot_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snapshot",
                "state",
                "cos_theta",
                "GS_force_file",
                "ES_force_file"
            ]
        )
        writer.writeheader()

        for row in sorted(by_snapshot_rows, key=lambda r: (r["snapshot"], state_sort_key(r["state"]))):
            writer.writerow(row)

    # Compute average and sample variance for each excited state.
    summary_rows = []

    for state in sorted(states, key=state_sort_key):
        vals = [row["cos_theta"] for row in by_snapshot_rows if row["state"] == state]

        if len(vals) == 0:
            continue

        summary_rows.append({
            "state": state,
            "n_geometries": len(vals),
            "average_cos_theta": average(vals),
            "variance_cos_theta": sample_variance(vals)
        })

    summary_file = args.out_prefix + "_summary.csv"

    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "state",
                "n_geometries",
                "average_cos_theta",
                "variance_cos_theta"
            ]
        )
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    # Write missing/problem files.
    missing_file = args.out_prefix + "_missing_files.csv"

    with open(missing_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snapshot",
                "state",
                "problem"
            ]
        )
        writer.writeheader()

        for row in missing_rows:
            writer.writerow(row)

    print()
    print("=== Table for SI ===")
    print("state,n_geometries,average_cos_theta,variance_cos_theta")

    for row in summary_rows:
        print(
            f"{row['state']},"
            f"{row['n_geometries']},"
            f"{row['average_cos_theta']:.8f},"
            f"{row['variance_cos_theta']:.8e}"
        )

    print()
    print("=== Output files ===")
    print(by_snapshot_file)
    print(summary_file)
    print(missing_file)

    if len(missing_rows) > 0:
        print()
        print("WARNING: Some files are missing or problematic.")
        print("Check:", missing_file)


if __name__ == "__main__":
    main()