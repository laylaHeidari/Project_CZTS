\# Bootstrap Uncertainty Analysis



\## What It Does



This notebook estimates uncertainty in fitted recombination times by bootstrap resampling over independent initial-condition population traces.



Each trial resamples the initial conditions, averages the selected traces, refits the recovery model, and contributes one recombination time to the 95% confidence interval.



\## Why It Matters



A narrow confidence interval indicates a more stable fitted lifetime, while a broad interval indicates stronger sensitivity to initial-condition sampling.



\## Required Libraries



The notebook imports `numpy`, `matplotlib`, `h5py`, `scipy`, `csv`, and `pathlib`.



\## Inputs



Each method and initial condition must contain a `mem\_data.hdf` file inside a folder named like `METHOD\_NBRA\_icond\_INDEX`.



The HDF5 file must contain the datasets `time/data` and `sh\_pop\_adi/data`.



\## Run



Open `bootstrap\_recombination\_uncertainty.ipynb`, place it beside the method folders, update the settings in the first code cell, and run all cells.



\## Outputs



The notebook creates `recombination\_bootstrap.pdf`, `recombination\_bootstrap.png`, and `recombination\_bootstrap\_summary.csv`.

