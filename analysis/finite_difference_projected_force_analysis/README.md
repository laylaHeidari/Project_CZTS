# Finite-Difference Projected Force Analysis

## What This Analysis Does

This workflow estimates the excited-state force correction along selected local bond-distortion coordinates using plus and minus displaced TDDFPT calculations.
For excited state \(I\), the projected force correction relative to the ground state along coordinate \(Q\) is:

$$
\Delta F_{I,Q}=F_{I,Q}-F_{0,Q}
\approx -\frac{\partial \Omega_I}{\partial Q}
\approx -\frac{\Omega_I(Q+\Delta Q)-\Omega_I(Q-\Delta Q)}{2\Delta Q}
$$

Here, \(\Omega_I\) is the vertical excitation energy and \(\Delta Q\) is the displacement along the selected coordinate.

## Why It Is Important

The analysis provides a quantitative check of how strongly photoexcitation may alter nuclear forces along representative local coordinates.

It is useful for assessing the neglect of back-reaction approximation (NBRA), where nuclear motion is taken from a ground-state trajectory.

## How to Interpret the Results

- Smaller \(|\Delta F_{I,Q}|\) values indicate more similar excited-state and ground-state forces along that coordinate.
- Larger values indicate stronger excited-state sensitivity to that local distortion.
- The values are coordinate-specific projected corrections, not complete excited-state force vectors.
- This analysis does not replace full excited-state molecular dynamics.

## Required Inputs

- `czts_reference_structure.xyz` — reference atomic structure
- `generate_bond_displacements.py` — creates positive and negative displaced structures
- `cp2k_tddfpt_template.inp` — CP2K TDDFPT input template
- completed CP2K outputs for every positive and negative displacement

## How to Run

Generate the displaced structures:


python generate_bond_displacements.py


Run the generated positive and negative TDDFPT calculations with CP2K.

Analyze the completed outputs:


python analyze_projected_force_corrections.py


## Output Files


results/projected_force_results.csv
results/projected_force_statistics.csv


`projected_force_results.csv` contains the correction for every state and coordinate.

`projected_force_statistics.csv` summarizes the minimum, mean, median, and maximum absolute correction for each excited state.
