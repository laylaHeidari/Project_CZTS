Direct Ground- and Excited-State Force Analysis

This folder contains matched CP2K inputs for calculating atomic forces for the CZTS ground state and the first singlet excited state, S1.

Both calculations use the same CZTS geometry, periodic cell, PBE functional, D3 dispersion correction, basis sets, pseudopotentials, and Cu Hubbard correction of U(Cu) = 6 eV. The S1 calculation additionally uses TDDFPT to obtain the analytic excited-state force.

## Force Difference

The excitation-induced atomic force change is obtained from

ΔF = F(S1) - F(GS)

The two force files must be compared atom by atom and in the same units. The force-file header should be checked before subtraction.

## Structural Interpretation

The Cartesian force difference can be projected onto structural coordinates to determine which distortions are favored after excitation. Examples include:

- Cu-S, Zn-S, and Sn-S bond-length coordinates
- S-metal-S and metal-S-metal bond-angle coordinates
- coordination-number collective variables

A positive or negative projected value indicates the preferred direction along the selected coordinate, while the magnitude indicates the strength of the excitation-induced driving force.

The CP2K inputs calculate Cartesian atomic forces. Bond, angle, and coordination-number projections require postprocessing of the force difference.

## Files

- `cp2k_gs_direct_force.inp`: ground-state force calculation
- `cp2k_s1_direct_force.inp`: S1 excited-state force calculation
- `czts_reference_structure.xyz`: common CZTS reference structure

## Run

Run both CP2K inputs from this directory using the same CP2K version and data files. Compare the resulting force files only after confirming that both calculations completed successfully.