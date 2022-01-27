import numpy as np
from scipy.special import erf

def get_combined_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp,suppression):
    """
    Combines properties of satellites and orphans

    Args:
        halo_data (dict): dict containing properties of host and subhalos
        params (dict): dict containing free parameters
        hparams (dict): dict containing hyperparameters
        cosmo_params (dict): dict containing cosmological parameters
        vpeak_Mr_interp (interpolating function): implements Mv--Vpeak abundance matching relation

    Returns:
        combined_satellite_properties (dict): dict containing combined properties of satellites and orphans
    """
    satellite_properties = get_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp,suppression)
    orphan_satellite_properties = get_orphan_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp,suppression)
    combined_satellite_properties = {}
    for key in satellite_properties.keys():
        combined_satellite_properties[key] = np.concatenate((satellite_properties[key],orphan_satellite_properties[key]))
    return combined_satellite_properties


def cut_subhalo_catalog(halo_data,hparams,cosmo_params,orphan=False):
    """
    Applies resolution cuts to subhalo catalog

    Args:
        halo_data (dict): dict containing properties of host and subhalos
        hparams (dict): dict containing hyperparameters
        cosmo_params (dict): dict containing cosmological parameters
        orphan (Boolean): apply cuts to orphan (True) or non-orphan (False, default) satellites

    Returns:
        Halo_subs[cut_idx] (array): subhalo array with resolution cuts applied 
        cut_idx (array): indices of resolution cuts 
    """
    if orphan==False:
        Halo_subs = halo_data['Halo_subs']
        mpeak_idx = (Halo_subs['mpeak']*(1.-cosmo_params['omega_b']/cosmo_params['omega_m'])>hparams['mpeak_cut'])
        vpeak_idx = (Halo_subs['vpeak']>hparams['vpeak_cut'])
        vmax_idx = (Halo_subs['vmax']>hparams['vmax_cut'])
        cut_idx = mpeak_idx & vpeak_idx & vmax_idx
        return Halo_subs[cut_idx], cut_idx
    elif orphan==True:
        Halo_subs_orphan = halo_data['orphan_catalog']
        mpeak_idx_orphan = (halo_data['orphan_catalog_mpeak']*(1.-cosmo_params['omega_b']/cosmo_params['omega_m'])>hparams['mpeak_cut'])
        radii_idx_orphan = Halo_subs_orphan[:,3] < hparams['orphan_radii_cut']
        cut_idx_orphan = mpeak_idx_orphan & radii_idx_orphan
        return Halo_subs_orphan[cut_idx_orphan], cut_idx_orphan


def Mr_to_L(Mr,Mbol_sun=4.81):
    """
    Conversion from absolute magnitude to luminosity

    Args:
        Mr (array of floats): absolute magnitude values

    Returns:
        L (array of floats): luminosity valuees
    """
    L = 10**((-1.*Mr + Mbol_sun)/2.5)
    return L


def L_to_Mr(L,Mbol_sun=4.81):
    """
    Conversion from luminosity to absolute magnitude

    Args:
        L (array of floats): luminosity values

    Returns:
        Mr (array of floats): absolute magnitude values
    """
    Mr = -1.*(2.5*(np.log10(L))-Mbol_sun)
    return Mr


def Mr_to_mu(Mr,r12,pc_to_kpc=1000.):
    """
    Conversion from absolute magnitude and half-light radius to surface brightness

    Args:
        Mr (array of floats): absolute magnitude values
        r12 (array of floats): half-light radii (units of pc)

    Returns:
        mu (array of floats): surface brightness values
    """
    mu = Mr + 36.57+ 2.5*np.log10(2*np.pi*((r12/pc_to_kpc)**2))
    return mu


def draw_L(L,sigma_M):
    """
    Draw luminosity from lognormal distribution

    Args:
        L (array of floats): luminosity values
        sigma_M (float): scatter in luminosity at fixed Vpeak (in dex)

    Returns:
        L_draw (array of floats): luminosity values drawn from lognormal relation
    """
    L_draw = np.random.lognormal(np.log(L),(np.log(10)*sigma_M))
    return L_draw


def Vpeak_to_Mr(Vpeak,params,vpeak_Mr_interp):
    """
    Converts subhalo Vpeak to absolute magnitude

    Args:
        Vpeak (array of floats): subhalo Vpeak values
        params (dict): dict containing free parameters
        vpeak_Mr_interp (interpolating function): implements Mv--Vpeak abundance matching relation

    Returns:
        Mr (array of floats): absolute magnitude values
    """
    sort_idx = np.argsort(np.argsort(Vpeak))
    Mr_mean = vpeak_Mr_interp(Vpeak,params['alpha'])[sort_idx]
    L_mean = Mr_to_L(Mr_mean)
    L = draw_L(L_mean,params['sigma_M'])
    Mr = L_to_Mr(L)
    return Mr


def get_halocentric_positions(Halo_main,Halo_subs_cut,hparams,cosmo_params,Mpc_to_kpc=1000.):
    """
    Gets halocentric radii and Cartesian coordinates of subhalos

    Args:
        Halo_main (array of floats): properties of main halo
        Halo_subs_cut (array of arrays): array of subhalo properties
        hparams (dict): dict containing hyperparameters
        cosmo_params (dict): dict containing cosmological parameters

    Returns:
        radii (array): subhalo halocentric radii in kpc, scaled according to hparam['chi']
        pos (array): subhalo halocentric Cartesian coordinates in kpc
    """
    Halox = (Halo_subs_cut['x']-Halo_main[0]['x'])*(Mpc_to_kpc/cosmo_params['h'])
    Haloy = (Halo_subs_cut['y']-Halo_main[0]['y'])*(Mpc_to_kpc/cosmo_params['h'])
    Haloz = (Halo_subs_cut['z']-Halo_main[0]['z'])*(Mpc_to_kpc/cosmo_params['h'])
    radii = np.sqrt(Halox**2 + Haloy**2 + Haloz**2)*hparams['chi']
    pos = np.vstack((Halox,Haloy,Haloz)).T
    return radii, pos


def get_satellite_sizes(rvir_acc,rs_acc,Halo_subs_cut,params,hparams,cosmo_params,c_normalization=10.0):
    """
    Gets half-light radii of satellites 

    Args:
        rvir_acc (array of floats): virial radii of subhalos at accretion
        rs_acc (array of floats): NFW scale radii of subhalos at accretion
        Halo_subs_cut (array of arrays): array of subhalo properties
        params (dict): dict containing free parameters
        hparams (dict): dict containing hyperparameters
        cosmo_params (dict): dict containing cosmological parameters
        c_normalization (float): concentration normalization in Jiang+ size relation

    Returns:
        r12 (array): half-light radii of satellites in pc
    """
    c_acc = rvir_acc/rs_acc
    c_correction = (c_acc/c_normalization)**(hparams['gamma_r'])
    beta_correction = ((Halo_subs_cut['vmax']/Halo_subs_cut['vacc']).clip(max=1.0))**hparams['beta']
    Halo_r12 = params['A']*c_correction*beta_correction*((rvir_acc/(hparams['R0']*cosmo_params['h']))**params['n'])
    r12 = np.random.lognormal(np.log(Halo_r12),np.log(10)*params['sigma_r'])
    return r12


def occupation_fraction(Mpeak,params,cosmo_params):
    """
    Gets satellite occupation fraction 

    Args:
        Mpeak (array of floats): peak virial mass of subhalos
        params (dict): dict containing free parameters
        cosmo_params (dict): dict containing cosmological parameters

    Returns:
        fgal (array of floats): probability that each subhalo hosts a satellite
    """
    fgal = (0.5*(1.+erf((np.log10((Mpeak/cosmo_params['h'])*(1-cosmo_params['omega_b']/cosmo_params['omega_m']))-params['M50'])/(np.sqrt(2)*params['sigma_mpeak'])))).clip(max=1.)
    return fgal


def wdm_shmf_suppression(Mpeak,params,cosmo_params,lovell_new=False,lovell_alt=False,schneider=False):
	if lovell_new==True:
		return (1+(4.2*(10**params['Mhm'])/(Mpeak/cosmo_params['h']))**2.5)**-0.2

	elif lovell_alt==True:
		return (1+(10**params['Mhm'])/(Mpeak/cosmo_params['h']))**-1.3

	elif schneider==True:
		return (1+(10**params['Mhm'])/(Mpeak/cosmo_params['h']))**-1.16

	else:
		return (1+2.7*(10**params['Mhm'])/(Mpeak/cosmo_params['h']))**-0.99


def get_survival_prob(ML_prob,Mpeak,params,cosmo_params,suppression):
    """
    Gets satellite survival probability

    Args:
        ML_prob (array of floats): subhalo disruption probability due to Galactic disk
        Mpeak (array of floats): peak virial mass of subhalos
        params (dict): dict containing free parameters
        cosmo_params (dict): dict containing cosmological parameters

    Returns:
        prob (array of floats): survival probability of each satellite (survival probability due to disk x occupation fraction)
    """
    baryonic_disruption_prob = ML_prob**(1./params['B'])
    occupation_prob = occupation_fraction(Mpeak,params,cosmo_params)
    prob = (1.-baryonic_disruption_prob)*occupation_prob

    if suppression == 'wdm':
    	prob *= wdm_shmf_suppression(Mpeak,params,cosmo_params)

    return prob


def get_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp,suppression):
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
    halo_data['Halo_subs_cut'], cut_idx = cut_subhalo_catalog(halo_data,hparams,cosmo_params)

    #Calculate luminosities
    satellite_properties['Mr'] = Vpeak_to_Mr(halo_data['Halo_subs_cut']['vpeak'],params,vpeak_Mr_interp)

    #Calculate positions
    satellite_properties['radii'], satellite_properties['pos'] = get_halocentric_positions(halo_data['Halo_main'],halo_data['Halo_subs_cut'],hparams,cosmo_params)

    #Calculate sizes
    satellite_properties['r12'] = get_satellite_sizes(halo_data['rvir'][cut_idx],halo_data['rs'][cut_idx],halo_data['Halo_subs_cut'],params,hparams,cosmo_params)
    satellite_properties['mu'] = Mr_to_mu(satellite_properties['Mr'],satellite_properties['r12'])

    #Calculate disruption probability due to baryonic effects and occupation fraction
    satellite_properties['prob'] = get_survival_prob(halo_data['Halo_ML_prob'][cut_idx],halo_data['Halo_subs_cut']['mpeak'],params,cosmo_params,suppression)

    return satellite_properties


def get_orphan_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp,suppression):
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

        vpeak_Mr_interp (interpolating function): implements Mv--Vpeak abundance matching relation

    Returns:
        orphan_satellite_properties (dict): dict containing r-band absolute magnitudes, Galactocentric radii, half-light radii, disruption probabilities, and 3D Galactocentric positions of luminous satellites
    """
    #Define output dict
    orphan_satellite_properties = {}

    #Cut subhalo catalogs
    Halo_subs_cut, cut_idx = cut_subhalo_catalog(halo_data,hparams,cosmo_params,orphan=True)

    #Calculate luminosities
    orphan_satellite_properties['Mr'] = Vpeak_to_Mr(Halo_subs_cut[:,5],params,vpeak_Mr_interp)

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
    orphan_satellite_properties['mu'] = Mr_to_mu(orphan_satellite_properties['Mr'],orphan_satellite_properties['r12'])

    #Calculate disruption probability due to baryonic effects and occupation fraction
    baryonic_disruption_prob = (1.-halo_data['orphan_aacc'][cut_idx])**(hparams['O'])
    occupation_prob = (0.5*(1.+erf((np.log10((halo_data['orphan_catalog_mpeak'][cut_idx]/cosmo_params['h'])*(1-cosmo_params['omega_b']/cosmo_params['omega_m']))-params['M50'])/(np.sqrt(2)*params['sigma_mpeak'])))).clip(max=1.)
    orphan_satellite_properties['prob'] = (1.-baryonic_disruption_prob)*occupation_prob
    if suppression=='wdm':
    	orphan_satellite_properties['prob'] *= wdm_shmf_suppression(halo_data['orphan_catalog_mpeak'][cut_idx],params,cosmo_params)

    return orphan_satellite_properties
