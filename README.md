# subhalo_satellite_connection
Code releases and accompanying papers:

### [1809.05542](https://arxiv.org/abs/1809.05542): Modeling the Connection Between Subhalos and Satellites in Milky Way-Like Systems. 

-- Implements the satellite galaxy--subhalo connection model described in the papers, which maps subhalos in dark matter-only zoom-in simulations of MW-mass halos to their corresponding satellite galaxies.

-- Contains tools for forward-modeling the population of classical and SDSS MW satellites.

### [1912.03303](https://arxiv.org/abs/1912.03303): Milky Way Satellite Census. II. Galaxy-Halo Connection Constraints Including the Impact of the Large Magellanic Cloud.

-- Implements the updated satellite galaxy--subhalo connection model described in the papers, which builds on the V1 model by including tools to analyze simulations with realistic Large Magellanic Cloud analogs. 

-- Contains tools for forward-modeling the population of DES and Pan-STARRS MW satellites.

-- The healpix masks necessary to implement the DES and PS1 survey selection functions in V2 can downloaded using the following links. Please see https://github.com/des-science/mw-sats for additional information.

DES mask: https://drive.google.com/file/d/1yF0pysai6rY13x17YY3KXZGVeDp25skz/view?usp=sharing
  
PS1 mask: https://drive.google.com/file/d/1UBlFSKB5CehFy1lwrITKRmoiFWvs3cK3/view?usp=sharing

### [2401.10318](https://arxiv.org/abs/2401.10318): Forecasts for Galaxy Formation and Dark Matter Constraints from Dwarf Galaxy Surveys.

**Under construction**

-- Implements the satellite galaxy--subhalo connection model and inference evaluated on mock halo catalogs, for various galaxy formation, WDM, and SHMF models

-- Improves vectorization of MCMC evaluation for MPI runs

### Utils

-- Contains tools for calculating the orbits of "orphan" (disrupted) subhalos in dark matter-only simulations, which are used in the V1 and V2 analyses, and tools for computing the likelihood of a satellite galaxy--subhalo connection model given an observed satellite population.

### Data

-- Contains zoom-in subhalo catalogs, abundance matching interpolator, and Milky Way satellite data

### Classifier

-- Contains XGBoost satellite selection function model from [1912.03302](https://arxiv.org/abs/1912.03302).
