import os
import matplotlib.pyplot as plt   # plots
import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit
import h5py
import warnings
import time

from liblibra_core import *
import util.libutil as comn
from libra_py import units, data_conv, dynamics_plotting
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.dynamics.tsh.plot as tsh_dynamics_plot
import libra_py.data_savers as data_savers
import libra_py.workflows.nbra.decoherence_times as decoherence_times

import multiprocessing as mp
from recipes import dish_nbra, fssh2_nbra, mash_nbra, fssh_nbra, gfsh_nbra, ida_nbra, msdm_nbra

warnings.filterwarnings('ignore')

colors = {}
colors.update({"11": "#8b1a0e"})
colors.update({"12": "#FF4500"})
colors.update({"13": "#B22222"})
colors.update({"14": "#DC143C"})
colors.update({"21": "#5e9c36"})
colors.update({"22": "#006400"})
colors.update({"23": "#228B22"})
colors.update({"24": "#808000"})
colors.update({"31": "#8A2BE2"})
colors.update({"32": "#00008B"})
colors.update({"41": "#2F4F4F"})

clrs_index = ["11", "21", "31", "41", "12", "22", "32", "13", "23", "14", "24"]

path_to_save_sd_Hvibs = '/path/to/res_occ_68_unocc_41'
istep = 1000
fstep = 4998
NSTEPS = fstep - istep

#================== Read energies =====================
E = []
for step in range(istep, fstep):
    energy_filename = F"{path_to_save_sd_Hvibs}/Hvib_ci_{step}_re.npz"
    energy_mat = sp.load_npz(energy_filename)
    E.append(np.array(np.diag(energy_mat.todense())))
E = np.array(E)
NSTATES = E[0].shape[0]

#================== Read time-overlap =====================
St = []
for step in range(istep, fstep):        
    St_filename = F"{path_to_save_sd_Hvibs}/St_ci_{step}_re.npz"
    St_mat = sp.load_npz(St_filename)
    St.append(np.array(St_mat.todense()))
St = np.array(St)

#================== Compute NACs and vibronic Hamiltonians =====================
NAC = []
Hvib = [] 
for c, step in enumerate(range(istep, fstep)):
    nac_filename = F"{path_to_save_sd_Hvibs}/Hvib_ci_{step}_im.npz"
    nac_mat = sp.load_npz(nac_filename)
    NAC.append(np.array(nac_mat.todense()))
    Hvib.append(np.diag(E[c, :])*(1.0 + 1j*0.0) - (0.0 + 1j)*nac_mat[:, :])
NAC = np.array(NAC)
Hvib = np.array(Hvib)

print('Number of steps:', NSTEPS)
print('Number of states:', NSTATES)
print(NAC.shape)
print(Hvib.shape)
print(St.shape)

#================== Decoherence times and energy gaps =====================
HAM_RE = []
for step in range(E.shape[0]):
    HAM_RE.append(data_conv.nparray2CMATRIX(np.diag(E[step, :])))
tau, rates = decoherence_times.decoherence_times_ave([HAM_RE], [0], NSTEPS, 0)
dE = decoherence_times.energy_gaps_ave([HAM_RE], [0], NSTEPS)
avg_deco = data_conv.MATRIX2nparray(tau) * units.au2fs
np.fill_diagonal(avg_deco, 0)
np.savetxt('avg_deco.txt', avg_deco.real)

gaps = MATRIX(NSTATES, NSTATES)
for step in range(NSTEPS):
    gaps += dE[step]
gaps /= NSTEPS

#================== PI suggestion: precompute CMATRIXs =====================
class tmp:
    pass

ham_adi_all = []
nac_adi_all = []
hvib_adi_all = []
u_all = []
st_all = []

for timestep in range(NSTEPS):
    ham_adi_all.append(data_conv.nparray2CMATRIX(np.diag(E[timestep, :])))
    nac_adi_all.append(data_conv.nparray2CMATRIX(NAC[timestep, :, :]))
    hvib_adi_all.append(data_conv.nparray2CMATRIX(Hvib[timestep, :, :]))
    x = CMATRIX(NSTATES, NSTATES); x.identity(); u_all.append(x)
    st_all.append(data_conv.nparray2CMATRIX(St[timestep, :, :]))

#================== Updated compute_model =====================
def compute_model(q, params, full_id):
    timestep = params["timestep"]
    obj = tmp()
    obj.ham_adi = ham_adi_all[timestep]
    obj.nac_adi = nac_adi_all[timestep]
    obj.hvib_adi = hvib_adi_all[timestep]
    obj.basis_transform = u_all[timestep]
    obj.time_overlap_adi = st_all[timestep]
    return obj

#================== Model parameters ====================
model_params = {"timestep": 0, "icond": 10, "model0": 0, "nstates": NSTATES}

#================== Dynamics settings ====================
dyn_general = {
    "nsteps": NSTEPS*2, "ntraj": 500, "nstates": NSTATES, "dt": 1.0*units.fs2au, "nfiles": NSTEPS,
    "decoherence_rates": rates, "ave_gaps": gaps,
    "progress_frequency": 0.1,
    "which_adi_states": range(NSTATES), "which_dia_states": range(NSTATES),
    "mem_output_level": 2,
    "properties_to_save": ["timestep", "time", "se_pop_adi", "sh_pop_adi"],
    "prefix": F"NBRA", "isNBRA": 0
}

dyn_general.update({"ham_update_method": 2})
dyn_general.update({"ham_transform_method": 0})
dyn_general.update({"time_overlap_method": 0})
dyn_general.update({"nac_update_method": 0})
dyn_general.update({"hvib_update_method": 0})
dyn_general.update({"force_method": 0, "rep_force": 1})
dyn_general.update({"hop_acceptance_algo": 32, "momenta_rescaling_algo": 0})
dyn_general.update({"rep_tdse": 1})
dyn_general.update({"electronic_integrator": 2})

#================== Load method =====================
fssh2_nbra.load(dyn_general); prf = "FSSH2"

#================== DOF and electronic init =====================
nucl_params = {"ndof": 1, "init_type": 3, "q": [-10.0], "p": [0.0], "mass": [2000.0], "force_constant": [0.01], "verbosity": -1}
istate = 118
elec_params = {"ndia": NSTATES, "nadi": NSTATES, "verbosity": -1, "init_dm_type": 0}
elec_params.update({"init_type": 1, "rep": 1, "istate": istate})

if prf == "MASH":
    istates = list(np.zeros(NSTATES))
    istates[istate] = 1.0
    elec_params.update({"init_type": 4, "rep": 1, "istate": istate, "istates": istates})

#================== Run a single job =====================
def function1(icond):
    print('Running the calculations for icond:', icond)
    rnd = Random()
    time.sleep(icond * 0.1)  # <-- PI suggestion
    dyn_general.update({"icond": icond})
    dyn_general.update({"prefix": F"{prf}_NBRA_icond_{icond}"})
    tsh_dynamics.generic_recipe(dyn_general, compute_model, model_params, elec_params, nucl_params, rnd)

################################
nthreads = 8
ICONDS = list(range(0, 4000, 400)) 
################################

pool = mp.Pool(nthreads)
pool.map(function1, ICONDS)
pool.close()
pool.join()

