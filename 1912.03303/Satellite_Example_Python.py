#Imports
import numpy as np
from load_hyperparams import load_hyperparams
from predict_satellites import *
from fix_lmc_coords import get_lmc_coords, rotate_about_LMC
from astropy.coordinates import SkyCoord
from masks import load_survey_masks, evaluate_mask
from ssf import apply_ssfs

#Set parameters
params = {}

params['alpha'] = -1.43
params['sigma_M'] = 0.05
params['M50'] = 7.5
params['sigma_mpeak'] = 0.05
params['B'] = 1.
params['A'] = 40
params['sigma_r'] = 0.6
params['n'] = 1.

#Load hyperparameters
hparams, cosmo_params, orphan_params, halo_data, vpeak_Mr_interp = load_hyperparams()

#Load survey masks and ssfs
surveys = ['des', 'ps1']

print('loading masks and ssfs ... \n')
masks,ssfs = load_survey_masks(surveys)
print('\n done')

#Return satellite properties for a particular MW-like host halo
combined_satellite_properties = get_combined_satellite_properties(halo_data[14], params, hparams, cosmo_params, 
                                                                  vpeak_Mr_interp)
    
#Rotate about LMC
lmc_ind = 0
lmc_cartesian_coords = get_lmc_coords(halo_data[14],cosmo_params,lmc_ind)
combined_satellite_properties_rotated = rotate_about_LMC(combined_satellite_properties,halo_data[14],cosmo_params,
                                                lmc_cartesian_coords,lmc_ind,observer_ind=0)

#Precompute satellite properties and flags for each observer location
combined_satellite_properties_list = []
DES_flags_list = []
PS1_flags_list = []
Halo_ra_list = []
Halo_dec_list = []

#Loop over simulations
for ind in [14, 20]:
    #Loop over observer locations
    for i in range(6):
        #Get satellite properties
        combined_satellite_properties = get_combined_satellite_properties(halo_data[ind], params, hparams, cosmo_params, 
                                                                  vpeak_Mr_interp)
        combined_satellite_properties_rotated = rotate_about_LMC(combined_satellite_properties,halo_data[ind],cosmo_params,
                                                    lmc_cartesian_coords,lmc_ind,i)
        #Transform to sky coordinates
        Halo_sky_coord = SkyCoord(x=combined_satellite_properties_rotated['rotated_pos'][:,0], 
                          y=combined_satellite_properties_rotated['rotated_pos'][:,1], 
                          z=combined_satellite_properties_rotated['rotated_pos'][:,2], 
                          unit='kpc', representation_type='cartesian').spherical
        combined_satellite_properties_rotated['ra'] = Halo_sky_coord.lon.degree
        combined_satellite_properties_rotated['dec'] = Halo_sky_coord.lat.degree
        #Assign flags 
        for survey in surveys:
            combined_satellite_properties_rotated['{}_flags'.format(survey)] = evaluate_mask(Halo_sky_coord.lon.degree,
                                                                                    Halo_sky_coord.lat.degree,
                                                                                    masks[survey], survey)
        #Apply ssfs
        combined_satellite_properties_rotated['pdet'] = apply_ssfs(combined_satellite_properties_rotated,ssfs)
        #Append
        combined_satellite_properties_list.append(combined_satellite_properties_rotated)

#First look at predictions
des_sum = []
ps1_sum = []

for i in range(len(combined_satellite_properties_list)):
    des_flags = combined_satellite_properties_list[i]['des_flags']
    ps1_flags = np.logical_and(combined_satellite_properties_list[i]['ps1_flags'],
                               ~combined_satellite_properties_list[i]['des_flags'])
    des_sum.append(np.sum(combined_satellite_properties_list[i]['pdet'][des_flags]))
    ps1_sum.append(np.sum(combined_satellite_properties_list[i]['pdet'][ps1_flags]))
    
print('number of predicted des satellites = {}'.format(np.mean(des_sum,axis=0)))
print('number of predicted ps1 satellites = {}'.format(np.mean(ps1_sum,axis=0)))