#Set paths
import sys
sys.path.append('../utils')

#Imports
import numpy as np
import emcee 
import progressbar
from load_hyperparams import load_params
from predict_satellites import get_combined_satellite_properties
from fix_lmc_coords import get_lmc_coords, rotate_about_LMC
from astropy.coordinates import SkyCoord
from masks import evaluate_mask
from ssf import apply_ssfs
from satellite_summary_stats import count_Mr_split
from stats import ln_marginalized_poisson_likelihood
from load_satellites import get_true_counts

def transform_to_sky_coords(rotated_pos):
    Halo_sky_coord = SkyCoord(x=rotated_pos[:,0],y=rotated_pos[:,1],z=rotated_pos[:,2],
                              unit='kpc',representation_type='cartesian').spherical
    return Halo_sky_coord.lon.degree, Halo_sky_coord.lat.degree


def satellite_realization(param_vector,hparams,cosmo_params,orphan_params,halo_data,sim_indices,vpeak_Mr_interp,masks,ssfs,suppression,surveys=['des','ps1']):
    #Set parameters
    params = load_params(param_vector)

    #Precompute satellite properties and flags for each observer location
    mock_counts = {}
    for survey in surveys:
      mock_counts['{}_bright_mu_split'.format(survey)] = []
      mock_counts['{}_dim_mu_split'.format(survey)] = []

    combined_satellite_properties_list = []
    DES_flags_list = []
    PS1_flags_list = []
    Halo_ra_list = []
    Halo_dec_list = []

    #Loop over simulations
    for i in range(len(sim_indices['host'])):
        #Loop over realizations of model at fixed parameters
        for j in range(hparams['n_realizations']):
            #Loop over observer locations
            #for k in range(6):
            k = np.random.randint(0,6)
            #Get satellite properties
            combined_satellite_properties = get_combined_satellite_properties(halo_data[sim_indices['host'][i]], 
                                                                                params, hparams, cosmo_params, vpeak_Mr_interp,suppression)
            #Rotate about LMC
            lmc_cartesian_coords = get_lmc_coords(halo_data[sim_indices['host'][i]],
                                                    cosmo_params,
                                                    sim_indices['LMC'][i])
            combined_satellite_properties_rotated = rotate_about_LMC(combined_satellite_properties,
                                                                        halo_data[sim_indices['host'][i]],
                                                                        cosmo_params,lmc_cartesian_coords,
                                                                        sim_indices['LMC'][i],k)
            #Transform to sky coordinates
            combined_satellite_properties_rotated['ra'], combined_satellite_properties_rotated['dec'] = transform_to_sky_coords(combined_satellite_properties_rotated['rotated_pos'])
            #Assign flags 
            for survey in surveys:
                combined_satellite_properties_rotated['{}_flags'.format(survey)] = evaluate_mask(combined_satellite_properties_rotated['ra'],
                                                                                            combined_satellite_properties_rotated['dec'],
                                                                                            masks[survey], survey)
            #Remove PS1 satellites that are also in DES
            if 'ps1' in surveys and 'des' in surveys:
                combined_satellite_properties_rotated['ps1_flags'][combined_satellite_properties_rotated['des_flags']==True] = False
                
            #Apply ssfs (loops over all surveys, so survey flags must yield unique object sets)
            combined_satellite_properties_rotated['pdet'] = apply_ssfs(combined_satellite_properties_rotated,ssfs,surveys=surveys)
            #Append satellite realizations
            combined_satellite_properties_list.append(combined_satellite_properties_rotated)
            #Append counts
            for survey in surveys:
                bright_mu_split, dim_mu_split = count_Mr_split(combined_satellite_properties_rotated['Mr'][combined_satellite_properties_rotated['{}_flags'.format(survey)]],
                                                                combined_satellite_properties_rotated['mu'][combined_satellite_properties_rotated['{}_flags'.format(survey)]],
                                                                combined_satellite_properties_rotated['pdet'][combined_satellite_properties_rotated['{}_flags'.format(survey)]])
                mock_counts['{}_bright_mu_split'.format(survey)].append(bright_mu_split)
                mock_counts['{}_dim_mu_split'.format(survey)].append(dim_mu_split)

    return combined_satellite_properties_list, mock_counts


def evaluate_ln_likelihood(param_vector,hparams,cosmo_params,orphan_params,halo_data,sim_indices,vpeak_Mr_interp,masks,ssfs,true_counts,
    suppression='cdm',surveys=['des','ps1']):
    true_counts = get_true_counts(surveys)
    combined_satellite_properties_list, mock_counts = satellite_realization(param_vector,hparams,cosmo_params,orphan_params,
                                                                            halo_data,sim_indices,vpeak_Mr_interp,masks,ssfs,suppression)

    ln_like = 0.

    for survey in surveys:
        ln_like += ln_marginalized_poisson_likelihood(true_counts['{}_bright_mu_split'.format(survey)],mock_counts['{}_bright_mu_split'.format(survey)])
        ln_like += ln_marginalized_poisson_likelihood(true_counts['{}_dim_mu_split'.format(survey)],mock_counts['{}_dim_mu_split'.format(survey)])

    return ln_like


def evaluate_ln_prior(param_vector,prior_hparams):
    prior_hparams_array = np.array(list(prior_hparams.values()))
    for i in range(len(param_vector)):
        if np.logical_and(param_vector[i]>prior_hparams_array[i][0],param_vector[i]<prior_hparams_array[i][1]):
            continue
        else:
            return -np.inf

    return np.log(1.0/(1.0+(param_vector[0]**2)))- ((np.log(param_vector[4]))**2)/(2*(0.5**2)) - ((param_vector[7]-1)**2)/(2*(0.5**2))


def evaluate_ln_post(param_vector,hparams,prior_hparams,cosmo_params,orphan_params,halo_data,sim_indices,vpeak_Mr_interp,masks,ssfs,true_counts,
    suppression='cdm',surveys=['des','ps1']):
    ln_prior = evaluate_ln_prior(param_vector,prior_hparams)
    if np.isinf(ln_prior):
        return ln_prior
    
    ln_like = evaluate_ln_likelihood(param_vector,hparams,cosmo_params,orphan_params,halo_data,sim_indices,vpeak_Mr_interp,masks,ssfs,true_counts,suppression,
    	surveys=['des','ps1'])
    ln_post = ln_like + ln_prior

    return ln_post


def mcmc(param_vector,hparams,prior_hparams,cosmo_params,orphan_params,halo_data,sim_indices,vpeak_Mr_interp,masks,ssfs,true_counts,suppression='cdm',surveys=['des','ps1']):
    ndim = len(param_vector)
    nwalkers = 4*ndim

    p0 = param_vector + 0.1*np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim))
    sampler = emcee.EnsembleSampler(nwalkers,ndim,evaluate_ln_post,args=[hparams,prior_hparams,cosmo_params,orphan_params,halo_data,sim_indices,vpeak_Mr_interp,
        masks,ssfs,true_counts,suppression,surveys])
    
    n_steps = 10000
    bar = progressbar.ProgressBar(n_steps).start()
    bar.update(0)

    for i,res in enumerate(sampler.sample(p0, iterations=n_steps)):
        bar.update(i+1)

    samples = sampler.chain[:, :, :].reshape((-1, ndim))

    return samples