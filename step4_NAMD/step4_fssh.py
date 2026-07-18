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


#from matplotlib.mlab import griddata
#%matplotlib inline 
warnings.filterwarnings('ignore')

colors = {}
colors.update({"11": "#8b1a0e"})  # red       
colors.update({"12": "#FF4500"})  # orangered 
colors.update({"13": "#B22222"})  # firebrick 
colors.update({"14": "#DC143C"})  # crimson   
colors.update({"21": "#5e9c36"})  # green
colors.update({"22": "#006400"})  # darkgreen  
colors.update({"23": "#228B22"})  # forestgreen
colors.update({"24": "#808000"})  # olive      
colors.update({"31": "#8A2BE2"})  # blueviolet
colors.update({"32": "#00008B"})  # darkblue  
colors.update({"41": "#2F4F4F"})  # darkslategray

clrs_index = ["11", "21", "31", "41", "12", "22", "32", "13","23", "14", "24"]


path_to_save_sd_Hvibs = '/path/to/res_occ_68_unocc_41'
istep = 1000
fstep = 4998
NSTEPS = fstep - istep
#================== Read energies =====================
E = []
for step in range(istep,fstep):
    energy_filename = F"{path_to_save_sd_Hvibs}/Hvib_ci_{step}_re.npz"
    energy_mat = sp.load_npz(energy_filename)
    # For data conversion we need to turn np.ndarray to np.array so that 
    # we can use data_conv.nparray2CMATRIX
    E.append( np.array( np.diag( energy_mat.todense() ) ) )
E = np.array(E)
NSTATES = E[0].shape[0]
#================== Read time-overlap =====================
St = []
for step in range(istep,fstep):        
    St_filename = F"{path_to_save_sd_Hvibs}/St_ci_{step}_re.npz"
    St_mat = sp.load_npz(St_filename)
    St.append( np.array( St_mat.todense() ) )
St = np.array(St)
#================ Compute NACs and vibronic Hamiltonians along the trajectory ============    
NAC = []
Hvib = [] 
for c, step in enumerate(range(istep,fstep)):
    nac_filename = F"{path_to_save_sd_Hvibs}/Hvib_ci_{step}_im.npz"
    nac_mat = sp.load_npz(nac_filename)
    NAC.append( np.array( nac_mat.todense() ) )
    Hvib.append( np.diag(E[c, :])*(1.0+1j*0.0)  - (0.0+1j)*nac_mat[:, :] )

NAC = np.array(NAC)
Hvib = np.array(Hvib)

print('Number of steps:', NSTEPS)
print('Number of states:', NSTATES)


print(NAC.shape)
print(Hvib.shape)
print(St.shape)



# ================= Computing the energy gaps and decoherence times
HAM_RE = []
for step in range(E.shape[0]):
    HAM_RE.append( data_conv.nparray2CMATRIX( np.diag(E[step, : ]) ) )
# Average decoherence times and rates
tau, rates = decoherence_times.decoherence_times_ave([HAM_RE], [0], NSTEPS, 0)
# Computes the energy gaps between all states for all steps
dE = decoherence_times.energy_gaps_ave([HAM_RE], [0], NSTEPS)
# Decoherence times in fs
avg_deco = data_conv.MATRIX2nparray(tau) * units.au2fs
# Zero all the diagonal elements of the decoherence matrix
np.fill_diagonal(avg_deco, 0)
# Saving the average decoherence times
np.savetxt('avg_deco.txt',avg_deco.real)
# Computing the average energy gaps
gaps = MATRIX(NSTATES, NSTATES)
for step in range(NSTEPS):
    gaps += dE[step]
gaps /= NSTEPS


class tmp:
    pass


ham_adi_all = []
nac_adi_all = []
hvib_adi_all = []
u_all = []
st_all = []

for timestep in range(NSTEPS):
    ham_adi_all.append( data_conv.nparray2CMATRIX( np.diag(E[timestep, : ]) ) )
    nac_adi_all.append( data_conv.nparray2CMATRIX( NAC[timestep, :, :] ) )
    hvib_adi_all.append(  data_conv.nparray2CMATRIX( Hvib[timestep, :, :] ) )
    x = CMATRIX(NSTATES, NSTATES); x.identity(); u_all.append( x )
    st_all.append( data_conv.nparray2CMATRIX( St[timestep, :, :] ) )

def compute_model(q, params, full_id):
    timestep = params["timestep"]
    nst = params["nstates"]
    obj = tmp()

    obj.ham_adi =  ham_adi_all[timestep]  #data_conv.nparray2CMATRIX( np.diag(E[timestep, : ]) )
    obj.nac_adi = nac_adi_all[timestep] #data_conv.nparray2CMATRIX( NAC[timestep, :, :] )
    obj.hvib_adi = hvib_adi_all[timestep] #data_conv.nparray2CMATRIX( Hvib[timestep, :, :] )
    obj.basis_transform = u_all[timestep] #CMATRIX(nst,nst); obj.basis_transform.identity()  #basis_transform
    obj.time_overlap_adi = st_all[timestep] #data_conv.nparray2CMATRIX( St[timestep, :, :] )
    
    return obj


#================== Model parameters ====================
model_params = { "timestep":0, "icond":10,  "model0":0, "nstates":NSTATES }




#=============== Some automatic variables, related to the settings above ===================

dyn_general = { "nsteps":NSTEPS*2, "ntraj":500, "nstates":NSTATES, "dt":1.0*units.fs2au, "nfiles": NSTEPS,
                "decoherence_rates":rates, "ave_gaps": gaps,
                "progress_frequency":0.1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),
                "mem_output_level":2,
                "properties_to_save":[ "timestep", "time","se_pop_adi", "sh_pop_adi" ],
                "prefix":F"NBRA", "isNBRA":0
              }

#=========== Some NBRA-specific parameters - these are the key settings for running file-based NBRA calculations ===========

dyn_general.update({"ham_update_method":2})  # read adiabatic properties from mthe files

dyn_general.update({"ham_transform_method":0})  # don't attempt to compute adiabatic properties from the diabatic ones, not to
                                                # override the read ones 
    
dyn_general.update({"time_overlap_method":0})  # don't attempt to compute those, not to override the read ones

dyn_general.update({"nac_update_method":0})    # don't attempt to recompute NACs, so that we don't override the read values

dyn_general.update({"hvib_update_method":0})   # don't attempt to recompute Hvib, so that we don't override the read values


dyn_general.update( {"force_method":0, "rep_force":1} ) # NBRA = don't compute forces, so rep_force actually doesn't matter


dyn_general.update({"hop_acceptance_algo":32, "momenta_rescaling_algo":0 })  # accept based on Boltzmann, no velocity rescaling

dyn_general.update( {"rep_tdse":1}) # the TDSE integration is conducted in adiabatic rep

dyn_general.update( {"electronic_integrator":2} )  # using the local diabatization approach to integrate TD-SE


#=================== Dynamics =======================
# Nuclear DOF - these parameters don't matter much in the NBRA calculations
nucl_params = {"ndof":1, "init_type":3, "q":[-10.0], "p":[0.0], "mass":[2000.0], "force_constant":[0.01], "verbosity":-1 }

# Amplitudes are sampled
elec_params = {"ndia":NSTATES, "nadi":NSTATES, "verbosity":-1, "init_dm_type":0}


elec_params.update( {"init_type":1,  "rep":1,  "istate":10 } )  # how to initialize: random phase, adiabatic representa


#============ Surface hopping opntions =================
##########################################################
#============== Select the method =====================
#dish_nbra.load(dyn_general); prf = "DISH"  # DISH
fssh_nbra.load(dyn_general); prf = "FSSH"  # FSSH
#fssh2_nbra.load(dyn_general); prf = "FSSH2"  # FSSH2
#gfsh_nbra.load(dyn_general); prf = "GFSH"  # GFSH
#ida_nbra.load(dyn_general); prf = "IDA"  # IDA
#mash_nbra.load(dyn_general); prf = "MASH"  # MASH
#msdm_nbra.load(dyn_general); prf = "MSDM"  # MSDM
##########################################################

#=================== Initial conditions =======================
#============== Nuclear DOF: these parameters don't matter much in the NBRA calculations ===============
nucl_params = {"ndof":1, "init_type":3, "q":[-10.0], "p":[0.0], "mass":[2000.0], "force_constant":[0.01], "verbosity":-1 }

#============== Electronic DOF: Amplitudes are sampled ========
elec_params = {"ndia":NSTATES, "nadi":NSTATES, "verbosity":-1, "init_dm_type":0}

###########
istate = 118
###########
elec_params.update( {"init_type":1,  "rep":1,  "istate":istate } )  # how to initialize: random phase, adiabatic representation

if prf=="MASH":
    istates = list(np.zeros(NSTATES))
    istates[istate] = 1.0
    elec_params.update( {"init_type":4,  "rep":1,  "istate":istate, "istates":istates } )  # different initialization for MASH





#for icond in range(0,4000,200):
def function1(icond):
    print('Running the calculations for icond:', icond)
    rnd = Random()
    time.sleep( icond * 0.1 )
    print('Running the calculations for icond:', icond)
    dyn_general.update({"icond": icond})
    dyn_general.update({"prefix":F"{prf}_NBRA_icond_{icond}"})
    res = tsh_dynamics.generic_recipe(dyn_general, compute_model, model_params, elec_params, nucl_params, rnd)


################################
nthreads = 8
ICONDS = list(range(0,4000,400)) 
################################

pool = mp.Pool(nthreads)
pool.map(function1, ICONDS)
pool.close()                            
pool.join()






