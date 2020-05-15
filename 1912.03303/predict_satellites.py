import numpy as np
from scipy.special import erf

def get_combined_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp):
    satellite_properties = get_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp)
    orphan_satellite_properties = get_orphan_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp)
    combined_satellite_properties = combine_satellite_properties(satellite_properties,orphan_satellite_properties)
    return combined_satellite_properties


def combine_satellite_properties(satellite_properties,orphan_satellite_properties):
    combined_satellite_properties = {}
    for key in satellite_properties.keys():
        combined_satellite_properties[key] = np.concatenate((satellite_properties[key],orphan_satellite_properties[key]))
    return combined_satellite_properties


def cut_subhalo_catalog(Halo_subs,hparams,cosmo_params):
    mpeak_idx = (Halo_subs['mpeak']*(1.-cosmo_params['omega_b']/cosmo_params['omega_m'])>hparams['mpeak_cut'])
    vpeak_idx = (Halo_subs['vpeak']>hparams['vpeak_cut'])
    vmax_idx = (Halo_subs['vmax']>hparams['vmax_cut'])
    cut_idx = mpeak_idx & vpeak_idx & vmax_idx
    return Halo_subs[cut_idx], cut_idx


def cut_orphan_subhalo_catalog(halo_data,hparams,cosmo_params):
    Halo_subs = halo_data['orphan_catalog']
    mpeak_idx = (halo_data['orphan_catalog_mpeak']*(1.-cosmo_params['omega_b']/cosmo_params['omega_m'])>hparams['mpeak_cut'])
    radii_idx = Halo_subs[:,3] < hparams['orphan_radii_cut']
    cut_idx = mpeak_idx & radii_idx
    return Halo_subs[cut_idx], cut_idx


def Mr_to_L(Mr,Mbol_sun=4.81):
    return 10**((-1.*Mr + Mbol_sun)/2.5)


def L_to_Mr(L,Mbol_sun=4.81):
    return -1.*(2.5*(np.log10(L))-Mbol_sun)


def draw_L(L,sigma_M):
    return np.random.lognormal(np.log(L),(np.log(10)*sigma_M))


def luminosity_from_vpeak(Vpeak,params,vpeak_Mr_interp):
    sort_idx = np.argsort(np.argsort(Vpeak))
    Mr_mean = vpeak_Mr_interp(Vpeak,params['alpha'])[sort_idx]
    L_mean = Mr_to_L(Mr_mean)
    L = draw_L(L_mean,params['sigma_M'])
    return L_to_Mr(L)


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
    Halo_subs_cut, cut_idx = cut_subhalo_catalog(halo_data['Halo_subs'],hparams,cosmo_params)
    #Calculate luminosities
    satellite_properties['Mr'] = luminosity_from_vpeak(Halo_subs_cut['vpeak'],params,vpeak_Mr_interp)
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


def get_orphan_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp):
    """
    Returns properties of luminous orphan satellites corresponding to disrupted subhalos that have been tracked to z=0 in a DMO zoom-in simulation

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

        vpeak_Mr_interp (interpolating function): implements Mv--Vpeak abund        ance matching relation

    Returns:
    orphan_satellite_properties (dict): dict containing r-band absolute         magnitudes, Galactocentric radii, half-light radii, disruption proba        bilities, and 3D Galactocentric positions of luminous satellites
    """
    #Define output dict
    orphan_satellite_properties = {}
    #Cut subhalo catalogs
    Halo_subs_cut, cut_idx = cut_orphan_subhalo_catalog(halo_data,hparams,cosmo_params)
    #Calculate luminosities
    orphan_satellite_properties['Mr'] = luminosity_from_vpeak(Halo_subs_cut[:,5],params,vpeak_Mr_interp)
    #Calculate positions
    Halo_main = halo_data['Halo_main']
    Halox = hparams['chi']*(Halo_subs_cut[:,0]-(Halo_main[0]['x']*1000/cosmo_params['h']))
    Haloy = hparams['chi']*(Halo_subs_cut[:,1]-(Halo_main[0]['y']*1000/cosmo_params['h']))
    Haloz = hparams['chi']*(Halo_subs_cut[:,2]-(Halo_main[0]['z']*1000/cosmo_params['h']))
    orphan_satellite_properties['radii'] = Halo_subs_cut[:,3]*hparams['chi']
    orphan_satellite_properties['pos'] = np.vstack((Halox,Haloy,Haloz)).T
    #Calculate sizes
    c = halo_data['rvir_orphan']/halo_data['rs_orphan']
    c_correction = (c[cut_idx]/10.0)**(hparams['gamma_r'])
    beta_correction = ((Halo_subs_cut[:,4]/Halo_subs_cut[:,9]).clip(max=1.0))**hparams['beta']
    Halo_r12 = params['A']*c_correction*beta_correction*((halo_data['rvir_orphan'][cut_idx]/(hparams['R0']*0.702))**params['n'])
    orphan_satellite_properties['r12'] = np.random.lognormal(np.log(Halo_r12),np.log(10)*params['sigma_r'])
    #Calculate disruption probability due to baryonic effects and occupation fraction
    baryonic_disruption_prob = (1.-halo_data['orphan_aacc'][cut_idx])**(hparams['O'])
    occupation_prob = (0.5*(1.+erf((np.log10(halo_data['orphan_catalog_mpeak'][cut_idx]*(1-cosmo_params['omega_b']/cosmo_params['omega_m']))-params['M50'])/(np.sqrt(2)*params['sigma_mpeak'])))).clip(max=1.)
    orphan_satellite_properties['prob'] = 1.-((1.-baryonic_disruption_prob)*occupation_prob)   
    ###
    return orphan_satellite_properties