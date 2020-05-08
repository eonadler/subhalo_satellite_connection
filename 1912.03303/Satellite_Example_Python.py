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
satellite_properties = get_satellite_properties(halo_data[14], params, hparams, cosmo_params, vpeak_Mr_interp)

#Return orphan satellite properties for a particular host halo
orphan_satellite_properties = get_orphan_satellite_properties(halo_data[14], params, hparams, cosmo_params, vpeak_Mr_interp)

#Rotate about LMC
lmc_ind = 0
lmc_cartesian_coords = get_lmc_coords(halo_data[14],cosmo_params,lmc_ind)
satellite_properties_rotated = rotate_about_LMC(satellite_properties,halo_data[14],cosmo_params,
                                                lmc_cartesian_coords,lmc_ind)

#Precompute satellite properties and flags for each observer location
satellite_properties_list = []
DES_flags_list = []
PS1_flags_list = []
Halo_ra_list = []
Halo_dec_list = []

print('loading masks ...')
des_mask = load_mask('des')
ps1_mask = load_mask('ps1')
print('done')

for i in range(6):
    #Get satellite properties
    satellite_properties = get_satellite_properties(halo_data[14], params, hparams, cosmo_params, vpeak_Mr_interp)
    satellite_properties_rotated = rotate_about_LMC(satellite_properties,halo_data[14],cosmo_params,
                                                lmc_cartesian_coords,lmc_ind,i)
    #Transform to sky coordinates
    Halo_sky_coord = SkyCoord(x=satellite_properties_rotated['rotated_pos'][:,0], 
                      y=satellite_properties_rotated['rotated_pos'][:,1], 
                      z=satellite_properties_rotated['rotated_pos'][:,2], 
                      unit='kpc', representation_type='cartesian').spherical
    satellite_properties_rotated['ra'] = Halo_sky_coord.lon.degree
    satellite_properties_rotated['dec'] = Halo_sky_coord.lat.degree
    #Assign flags 
    satellite_properties_rotated['DES_flags'] = evaluate_mask(Halo_sky_coord.lon.degree,Halo_sky_coord.lat.degree,
                                                               des_mask)
    satellite_properties_rotated['PS1_flags'] = evaluate_mask(Halo_sky_coord.lon.degree,Halo_sky_coord.lat.degree,
                                                               ps1_mask)
    #Append
    satellite_properties_list.append(satellite_properties_rotated)