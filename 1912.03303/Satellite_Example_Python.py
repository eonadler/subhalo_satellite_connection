from load_hyperparams import load_hyperparams
from predict_satellites import satellite_properties
from predict_orphans import orphan_satellite_properties

#Set parameters
params = {}

params['alpha'] = -1.4
params['sigma_M'] = 0.1
params['M50'] = 8.
params['sigma_mpeak'] = 0.35
params['B'] = 1.
params['A'] = 34
params['sigma_r'] = 0.3
params['n'] = 1.3

#Load hyperparameters
hparams, cosmo_params, orphan_params, halo_data, vpeak_Mr_interp = load_hyperparams()

#Return satellite properties for a particular host halo
satellite_properties = satellite_properties(halo_data[0], params, hparams, cosmo_params, vpeak_Mr_interp)

#Return orphan satellite properties for a particular host halo
orphan_satellite_properties = orphan_satellite_properties(halo_data[0], params, hparams, cosmo_params, vpeak_Mr_interp)