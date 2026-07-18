# NTO Hole and Particle Analysis

## What This Script Does

"analyze_nto_states.py" analyzes hole and particle Natural Transition Orbital (NTO) cube files for excited states.

It calculates:

- element-resolved hole and particle percentages;
- hole and particle participation ratios; and
- the NTO-pair weights used for each state.

## Why It Is Needed

NTO cube files show where the hole and particle densities are located in real space, but they do not directly provide element percentages or localization measures.

This script converts the cube amplitudes to densities, assigns the density to the nearest atoms, and summarizes the results by element.

For a state represented by more than one NTO pair, the corresponding pair weights must be entered in `NTO_WEIGHTS` near the top of the script.

## Required Inputs

- matching hole and particle NTO `.cube` files;
- NTO-pair weights only for states containing more than one pair.

## How to Run

python analyze_nto_states.py


## Output Files

text
CZTS_NTO_states_summary.csv
CZTS_NTO_weights_used.csv


The first file contains the element percentages and participation ratios for each detected state. The second records the raw and normalized NTO-pair weights used in the analysis.
