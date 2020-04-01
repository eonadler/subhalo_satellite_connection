import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

def get_lmc_coords(halo_data,cosmo_params,lmc_true_sky_coords='05:23:34.5 -69:45.37'):
    """
    Returns Cartesian coordinates of LMC analog assuming (a) it is fixed to the sky position of the true LMC, and (b) the true LMC is at the distance of the mock LMC

    Args:
    halo_data (dict): dict containing properties of host and subhalos 
    cosmo_params (dict): dict containing cosmological parameters
    lmc_true_sky_coords (string): true RA and dec of LMC

    Returns:
    lmc_cartesian_coords (array): Cartesian coordinates of LMC analog
    """
	#Get MW and LMC analogs; currently assumes LMC analog is 0th subhalo
    mw_analog = halo_data['Halo_main'][0]
	lmc_analog = halo_data['Halo_subs'][0]

	#Get LMC distance and perform coordinate transformation
	lmc_distance = (1000./cosmo_params['h'])*np.sqrt((lmc_analog['x']-mw_analog['x'])**2+(lmc_analog['y']-mw_analog['y'])**2+(lmc_analog['z']-mw_analog['z'])**2)
	lmc_sky_coords = SkyCoord(lmc_true_sky_coords, unit=(u.hourangle, u.deg), distance=lmc_distance*u.kpc)
	lmc_cartesian_coords = np.array(lmc_sky_coords.cartesian.xyz)

	return lmc_cartesian_coords

def rotate_about_LMC(satellite_properties,lmc_cartesian_coords,observer_distance=8.,observer_locations=6):
    """
    Returns Cartesian coordinates of mock satellites, rotated such that the LMC is fixed at the Cartesian coordinates from the previous function

    Args:
    satellite_properties (dict): dict containing properties of satellites
    lmc_cartesian_coords (array): Cartesian coordinates of LMC analog

    Returns:
    satellite_properties (dict): dict containing properties of satellites including rotated positions
    """
	#Load satellite positions
	pos = satellite_properties['pos']
	rotated_pos = np.zeros((np.shape(pos)))

	#Change satellite positions to observer frame
	x = np.zeros((observer_locations,3))

	for i in range(observer_locations/2):
        x[i*2][i] = observer_distance
        x[i*2+1][i] = -1.*observer_distance

	pos = pos - x[np.random.randint(0,observer_locations)]

	#Set LMC coordinates (still assumes LMC analog is 0th subhalo)
	mock_lmc_coords = (pos[0][0], pos[0][1], pos[0][2])
	true_lmc_coords = lmc_cartesian_coords

	#Perform coordinate rotation
	a,b = (mock_lmc_coords/np.linalg.norm(mock_lmc_coords)), (true_lmc_coords/np.linalg.norm(true_lmc_coords))
    
	v = np.cross(a,b)
	c = np.dot(a,b)
	s = np.linalg.norm(v)
	I = np.identity(3)
	vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
	k = np.matrix(vXStr)
	r = I + k + np.matmul(k,k) * ((1 -c)/(s**2))

	for i in range (0,len(rotated_pos)):
        mock_lmc_coords = (rotated_pos[i][0],rotated_pos[i][1],rotated_pos[i][2])
        (rotated_pos[i][0],rotated_pos[i][1],rotated_pos[i][2]) = np.asarray(r*np.reshape(mock_lmc_coords,(3,1)))[:,0]

	satellite_properties['rotated_pos'] = rotated_pos

	return satellite_properties
