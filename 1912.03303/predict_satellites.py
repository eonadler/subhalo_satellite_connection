import numpy as np
from scipy.special import erf

def get_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp):
    """
    Returns properties of luminous satellites corresponding to surviving subhalos at z=0 in a DMO zoom-in simulation

    Args:
	halo_data (dict): dict containing properties of host and subhalos
 
        params (dict): dict containing free parameters
        params['alpha'] (float): faint-end slope of satellite luminosity function
		params['sigma_M'] (float): lognormal scatter in M_V--V_peak relation (in dex)
		params['M50'] (float): mass at which 50% of halos host galaxies (in log10(M*/h))
		params['sigma_mpeak'] (float): scatter in galaxy occupation fraction
		params['B'] (float): subhalo disruption probability (due to baryons)
		params['A']: satellite size relation amplitude
		params['sigma_r']: satellite size relation scatter
		params['n']: satellite size relation slope

        hparams (dict): dict containing hyperparameters
        hparams['vpeak_cut']: subhalo vpeak resolution threshold
        hparams['vmax_cut']: subhalo vmax resolution threshold
        hparams['chi']: satellite radial scaling
        hparams['R0']: satellite size relation normalization
        hparams['gamma_r']: slope of concentration dependence in satellite size relation
        hparams['beta']: tidal stripping parameter
        hparams['O']: orphan satellite parameter

        cosmo_params (dict): dict containing cosmological parameters
        cosmo_params['omega_b']: baryon fraction
        cosmo_params['omega_m']: matter fraction
        cosmo_params['h']: dimensionless hubble parameter

        vpeak_Mr_interp (interpolating function): implements Mv--Vpeak abundance matching relation

    Returns:
	satellite_properties (dict): dict containing r-band absolute magnitudes, Galactocentric radii, half-light radii, disruption probabilities, and 3D Galactocentric positions of luminous satellites
    """
    #Define output dict
    satellite_properties = {}
    #Cut subhalo catalogs
    Halo_subs = halo_data['Halo_subs']
    mpeak_idx = (Halo_subs['mpeak']*(1.-cosmo_params['omega_b']/cosmo_params['omega_m'])>10**7)
    vpeak_idx = (Halo_subs['vpeak']>hparams['vpeak_cut'])
    vmax_idx = (Halo_subs['vmax']>hparams['vmax_cut'])
    cut_idx = mpeak_idx & vpeak_idx & vmax_idx
    Halo_subs_cut = Halo_subs[cut_idx]
    #Calculate luminosities
    sort_idx = np.argsort(np.argsort(Halo_subs_cut['vpeak']))
    Mr_mean = vpeak_Mr_interp(Halo_subs_cut['vpeak'], params['alpha'])[sort_idx]
    L_mean = 10**((-1.*Mr_mean + 4.81)/2.5 + np.log10(2))
    L = np.random.lognormal(np.log(L_mean),(np.log(10)*params['sigma_M']))
    satellite_properties['Mr'] = -1.*(2.5*(np.log10(L)-np.log10(2))-4.81)
    #Calculate positions
    Halo_main = halo_data['Halo_main']
    Halox = hparams['chi']*(Halo_subs_cut['x']-Halo_main[0]['x'])*(1000/cosmo_params['h'])
    Haloy = hparams['chi']*(Halo_subs_cut['y']-Halo_main[0]['y'])*(1000/cosmo_params['h'])
    Haloz = hparams['chi']*(Halo_subs_cut['z']-Halo_main[0]['z'])*(1000/cosmo_params['h'])
    satellite_properties['radii'] = np.sqrt(Halox**2 + Haloy**2 + Haloz**2)*hparams['chi']
    satellite_properties['pos'] = np.vstack((Halox,Haloy,Haloz)).T
    #Calculate sizes
    c = halo_data['rvir']/halo_data['rs']
    c_correction = (c[cut_idx]/10.0)**(hparams['gamma_r'])
    beta_correction = ((Halo_subs_cut['vmax']/Halo_subs_cut['vacc']).clip(max=1.0))**hparams['beta']
    Halo_r12 = params['A']*c_correction*beta_correction*((Halo_subs_cut['rvir']/(hparams['R0']*0.702))**params['n'])
    satellite_properties['r12'] = np.random.lognormal(np.log(Halo_r12),np.log(10)*params['sigma_r'])
    #Calculate disruption probability due to baryonic effects and occupation fraction
    baryonic_disruption_prob = halo_data['Halo_ML_prob'][cut_idx]**(1./params['B'])
    occupation_prob = (0.5*(1.+erf((np.log10(Halo_subs_cut['mpeak']*(1-cosmo_params['omega_b']/cosmo_params['omega_m']))-params['M50'])/(np.sqrt(2)*params['sigma_mpeak'])))).clip(max=1.)
    satellite_properties['prob'] = 1.-((1.-baryonic_disruption_prob)*occupation_prob)
    ###
    return satellite_properties