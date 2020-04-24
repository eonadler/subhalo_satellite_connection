import numpy as np

def get_orphan_satellite_properties(halo_data,params,hparams,cosmo_params,vpeak_Mr_interp):
    """
    Returns properties of luminous orphan satellites corresponding to disrupted subhalos that have been tracked to z=0 in a DMO zoom-in simulation

    Args:
	halo_data (dict): dict containing properties of host and subhalos 

        params (dict): dict containing free parameters
        params['alpha'] (float): faint-end slope of satellite luminosity function
	params['sigma_M'] (float): lognormal scatter in M_V--V_peak relation (in dex)
	params['mpeak_cut'] (float): mass threshold for galaxy formation (in log10(M*/h))
	
        hparams (dict): dict containing hyperparameters
        hparams['vpeak_cut']: subhalo vpeak resolution threshold
        hparams['vmax_cut']: subhalo vmax resolution threshold
        hparams['sigma_M_min']: minimum allowed luminosity scatter
        hparams['chi']: satellite radial scaling
        hparams['A']: satellite size relation normalization
        hparams['gamma']: satellite size relation slope
        hparams['beta']: tidal stripping parameter
        hparams['sigma_r']: scatter in satellite size relation
        hparams['size_min']: minimum allowed satellite size
        hparams['O']: orphan satellite parameter

        cosmo_params (dict): dict containing cosmological parameters
        cosmo_params['omega_b']: baryon fraction
        cosmo_params['omega_m']: matter fraction
        cosmo_params['h']: dimensionless hubble parameter

        vpeak_Mr_interp (interpolating function): implements Mv--Vpeak abund        ance matching relation

    Returns:
	orphan_satellite_properties (dict): dict containing r-band absolute         magnitudes,	Galactocentric radii, half-light radii, disruption proba        bilities, and 3D Galactocentric positions of luminous satellites
    """
    #Define output dict
    orphan_satellite_properties = {}
    #Cut subhalo catalogs
    Halo_subs = halo_data['orphan_catalog']
    mpeak_idx = (halo_data['orphan_catalog_mpeak']*(1.-cosmo_params['omega_b']/cosmo_params['omega_m'])>10**(params['mpeak_cut']))
    radii_idx = Halo_subs[:,3] < 300
    cut_idx = mpeak_idx & radii_idx
    Halo_subs_cut = Halo_subs[cut_idx]
    #Calculate luminosities
    sort_idx = np.argsort(np.argsort(Halo_subs_cut[:,5]))
    Mr_mean = vpeak_Mr_interp(Halo_subs_cut[:,5], params['alpha'])[sort_idx]
    L_mean = 10**((-1.*Mr_mean + 4.81)/2.5 + np.log10(2))
    L = np.random.lognormal(np.log(L_mean),(np.log(10)*params['sigma_M']).clip(min=hparams['sigma_M_min']))
    orphan_satellite_properties['Mr'] = -1.*(2.5*(np.log10(L)-np.log10(2))-4.81)
    #Calculate positions
    Halo_main = halo_data['Halo_main']
    Halox = hparams['chi']*(Halo_subs_cut[:,0]-(Halo_main[0]['x']*1000/cosmo_params['h']))
    Haloy = hparams['chi']*(Halo_subs_cut[:,1]-(Halo_main[0]['y']*1000/cosmo_params['h']))
    Haloz = hparams['chi']*(Halo_subs_cut[:,2]-(Halo_main[0]['z']*1000/cosmo_params['h']))
    orphan_satellite_properties['radii'] = Halo_subs_cut[:,3]
    orphan_satellite_properties['pos'] = np.vstack((Halox,Haloy,Haloz)).T
    #Calculate sizes
    c = halo_data['rvir_orphan']/halo_data['rs_orphan']
    Halo_r12 = (hparams['A']*(c[cut_idx]/10.0)**(hparams['gamma'])*halo_data['rvir_orphan'][cut_idx]/cosmo_params['h'])**(((Halo_subs_cut[:,4]/Halo_subs_cut[:,9]).clip(max=1.0))**hparams['beta'])
    orphan_satellite_properties['r12'] = np.random.lognormal(np.log(Halo_r12),np.log(10)*hparams['sigma_r']).clip(min=hparams['size_min']) 
    #Calculate disruption probabilities
    orphan_satellite_properties['prob'] = (1.-halo_data['orphan_aacc'][cut_idx])**(hparams['O'])
    ###
    return orphan_satellite_properties
