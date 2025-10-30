import pickle
import numpy as np

def load_params(param_vector=[]):
    """
    Casts vector of parameters into dictionary
    """
    params = {}
    param_vector_default = [-1.43,0.05,7.5,0.05,1.,40.,0.6,1.,5.5]

    if len(param_vector) != 0:
        params['alpha'], params['sigma_M'], params['M50'], params['sigma_mpeak'], params['B'], params['A'], params['sigma_r'], params['n'], params['Mhm'] = param_vector
    else:
        params['alpha'], params['sigma_M'], params['M50'], params['sigma_mpeak'], params['B'], params['A'], params['sigma_r'], params['n'], params['Mhm'] = param_vector_default

    return params


def load_hyperparams():
    """
    Returns hyperparameters, cosmological parameters, and loads data for predict_satellites.py and predict_orphans.py
    """
    #Load halo data (encoding='latin1' for Python3)
    with open('../Data/halo_data.pkl', 'rb') as halo_input:
        halo_data = pickle.load(halo_input, encoding='latin1')

    #Load interpolator
    with open('../Data/interpolator.pkl', 'rb') as interp:
        vpeak_Mr_interp = pickle.load(interp, encoding='latin1')

    #Cosmological params
    cosmo_params = {}
    cosmo_params['omega_b'] = 0.0 
    cosmo_params['omega_m'] = 0.286
    cosmo_params['h'] = 0.7

    #hyperparameters
    hparams = {}
    hparams['mpeak_cut'] = 10**7
    hparams['vpeak_cut'] = 10.
    hparams['vmax_cut'] = 9.
    hparams['orphan_radii_cut'] = 300.
    hparams['chi'] = 1.
    hparams['R0'] = 10.0
    hparams['gamma_r'] = 0.0
    hparams['beta'] = 0.
    hparams['O'] = 1.
    hparams['n_realizations'] = 5

    #prior hyperparameters
    prior_hparams = {}
    prior_hparams['alpha'] = np.array([-2.,-1.1])
    prior_hparams['sigma_M'] = np.array([0.,2.])
    prior_hparams['M50'] = np.array([7.49,11.])
    prior_hparams['sigma_mpeak'] = np.array([1e-5,1.])
    prior_hparams['B'] = np.array([1e-5,3.])
    prior_hparams['A'] = np.array([10.,500.])
    prior_hparams['sigma_r'] = np.array([1e-5,2.])
    prior_hparams['n'] = np.array([0.,2.])
    prior_hparams['Mhm'] = np.array([5.,9.])

    #Orphan hyperparameters
    orphan_params = {}
    orphan_params['eps'] = 0.01 
    orphan_params['df'] = 1

    #Simulation and LMC indices
    sim_indices = {}
    sim_indices['host'] = [0,1]
    sim_indices['LMC'] = [0,0]

    return hparams, prior_hparams, cosmo_params, orphan_params, halo_data, sim_indices, vpeak_Mr_interp
