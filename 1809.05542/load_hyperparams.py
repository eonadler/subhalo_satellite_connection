import pickle

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
    hparams['vpeak_cut'] = 10.
    hparams['vmax_cut'] = 9.
    hparams['sigma_M_min'] = 10**-5
    hparams['chi'] = 1.
    hparams['A'] = 0.02
    hparams['gamma'] = -0.7
    hparams['beta'] = 0.
    hparams['sigma_r'] = 0.01
    hparams['size_min'] = 0.02
    hparams['O'] = 1.

    #Orphan hyperparameters
    orphan_params = {}
    orphan_params['eps'] = 0.01 
    orphan_params['df'] = 1
    ###
    return hparams, cosmo_params, orphan_params, halo_data, vpeak_Mr_interp