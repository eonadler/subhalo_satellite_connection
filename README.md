# subhalo_satellite_connection
Code releases and accompanying papers:

### [1809.05542](https://arxiv.org/abs/1809.05542): Modeling the Connection Between Subhalos and Satellites in Milky Way-Like Systems. 

-- Implements the satellite galaxy--subhalo connection model described in the papers, which maps subhalos in dark matter-only zoom-in simulations of MW-mass halos to their corresponding satellite galaxies.

-- Contains tools for forward-modeling the population of classical and SDSS MW satellites.

### [1912.03303](https://arxiv.org/abs/1912.03303): Milky Way Satellite Census. II. Galaxy-Halo Connection Constraints Including the Impact of the Large Magellanic Cloud.

-- Implements the updated satellite galaxy--subhalo connection model described in the papers, which builds on the V1 model by including tools to analyze simulations with realistic Large Magellanic Cloud analogs. 

-- Contains tools for forward-modeling the population of DES and Pan-STARRS MW satellites.

-- The healpix masks necessary to implement the DES and PS1 survey selection functions in V2 can downloaded using the following links. Please see https://github.com/des-science/mw-sats for additional information.

DES mask: [healpix_mask_des_v5.1.fits.zip](https://github.com/eonadler/subhalo_satellite_connection/releases/download/v0.4/healpix_mask_des_v5.1.fits.zip)
  
PS1 mask: [healpix_mask_ps1_v5.1.fits.zip](https://github.com/eonadler/subhalo_satellite_connection/releases/download/v0.4/healpix_mask_ps1_v5.1.fits.zip)

### [2401.10318](https://arxiv.org/abs/2401.10318): Forecasts for Galaxy Formation and Dark Matter Constraints from Dwarf Galaxy Surveys.

-- Implements the satellite galaxy--subhalo connection model and inference evaluated on mock halo catalogs, for various galaxy formation, WDM, and SHMF models

-- Improves vectorization of MCMC evaluation for MPI runs

### [2504.16203](https://arxiv.org/abs/2504.16203): Predictions for the Detectability of Milky Way Satellite Galaxies and Outer-Halo Star Clusters with the Vera C. Rubin Observatory.

-- Develops survey selection function and footprint for LSST.

-- Makes predictions for satellite population that will be observed by LSST.

LSST mask: [healpix_mask_lsst_v3.1.fits.zip](https://github.com/eonadler/subhalo_satellite_connection/releases/download/v0.4/healpix_mask_lsst_v3.1.fits.zip)


### Utils

-- Contains tools for calculating the orbits of "orphan" (disrupted) subhalos in dark matter-only simulations, which are used in the V1 and V2 analyses, and tools for computing the likelihood of a satellite galaxy--subhalo connection model given an observed satellite population.

### Data

-- Contains zoom-in subhalo catalogs, abundance matching interpolator, and Milky Way satellite data

### Classifier

-- Contains XGBoost satellite selection function models from [1912.03302](https://arxiv.org/abs/1912.03302) and [2504.16203](https://arxiv.org/abs/2504.16203).
