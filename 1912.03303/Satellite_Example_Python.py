from load_hyperparams import load_hyperparams
from predict_satellites import satellite_properties
from predict_orphans import orphan_satellite_properties
from fix_lmc_coords import get_lmc_coords, rotate_about_LMC

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
satellite_properties = get_satellite_properties(halo_data[0], params, hparams, cosmo_params, vpeak_Mr_interp)

#Return orphan satellite properties for a particular host halo
orphan_satellite_properties = get_orphan_satellite_properties(halo_data[0], params, hparams, cosmo_params, vpeak_Mr_interp)

#Rotate about LMC
lmc_ind = 0
lmc_cartesian_coords = get_lmc_coords(halo_data[0],cosmo_params,lmc_ind)
satellite_properties_rotated = rotate_about_LMC(satellite_properties,halo_data[0],cosmo_params,
                                                lmc_cartesian_coords,lmc_ind)