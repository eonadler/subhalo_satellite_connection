import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

def transform_to_observer_frame(pos,observer_ind,observer_distance=8.,observer_locations=6):
    """
    Transforms Galactocentric position vectors to the frame of a randomly oriented heliocentric observer at the correct Galactocentric distance

    Args:
    pos (array): array of Galactocentric satellite positions
    observer_distance: Galactocentric distance of sun (units of kpc)
    observer_locations: number of observer locations to choose from

    Returns:
    pos_observer_frame (array): array of satellite position vectors in heliocentric frame
    """
    x = np.zeros((observer_locations,3))

    for i in range(int(observer_locations/2)):
        x[i*2][i] = observer_distance
        x[i*2+1][i] = -1.*observer_distance
    
    pos_observer_frame = pos - x[observer_ind]

    return pos_observer_frame


def rotate_about_vector(pos,vec1,vec2):
    """
    Returns heliocentric satellite position vectors transformed via a rotation matrix that rotates the mock LMC analog into its correct sky position

    Args:
    pos (array): array of heliocentric satellite position vectors
    vec1 (array): original heliocentric coordinates of mock LMC
    vec2 (array): final heliocentric coordinates of mock LMC

    Returns:
    rotated_pos (array): array of position vectors in rotated frame
    """
    rotated_pos = np.zeros(np.shape(pos))

    a,b = (vec1/np.linalg.norm(vec1)), (vec2/np.linalg.norm(vec2))
    
    v = np.cross(a,b)
    c = np.dot(a,b)
    s = np.linalg.norm(v)
    I = np.identity(3)

    vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0., -v[2], v[1], v[2], 0., -v[0], -v[1], v[0], 0.)
    K = np.matrix(vXStr)
    R = I + K + np.matmul(K,K) * ((1.-c)/(s**2))

    for i in range (0,len(rotated_pos)):
        temp_vec = (pos[i][0],pos[i][1],pos[i][2])
        (rotated_pos[i][0],rotated_pos[i][1],rotated_pos[i][2]) = np.asarray(R*np.reshape(temp_vec,(3,1)))[:,0]

    return rotated_pos


def get_lmc_coords(halo_data,cosmo_params,lmc_ind,lmc_true_sky_coords='05:23:34.5 -69:45.37',Mpc_to_kpc=1000.):
    """
    Returns Cartesian coordinates of LMC analog assuming (a) it is fixed to the sky position of the true LMC, and (b) the true LMC is at the distance of the mock LMC

    Args:
    halo_data (dict): dict containing properties of host and subhalos 
    cosmo_params (dict): dict containing cosmological parameters
    lmc_ind (int): index of LMC analog in 'Halo_subs'
    lmc_true_sky_coords (string): true RA and dec of LMC

    Returns:
    lmc_cartesian_coords (array): Cartesian coordinates of LMC analog
    """
    #Get MW and LMC analogs
    mw_analog = halo_data['Halo_main'][0]
    lmc_analog = halo_data['Halo_subs'][lmc_ind]

    #Get LMC distance and perform coordinate transformation
    lmc_distance = (Mpc_to_kpc/cosmo_params['h'])*np.sqrt((lmc_analog['x']-mw_analog['x'])**2+(lmc_analog['y']-mw_analog['y'])**2+(lmc_analog['z']-mw_analog['z'])**2)
    lmc_sky_coords = SkyCoord(lmc_true_sky_coords,unit=(u.hourangle, u.deg),distance=lmc_distance*u.kpc)
    lmc_cartesian_coords = np.array(lmc_sky_coords.cartesian.xyz)

    return lmc_cartesian_coords


def rotate_about_LMC(satellite_properties,halo_data,cosmo_params,lmc_cartesian_coords,lmc_ind,observer_ind):
    """
    Returns Cartesian coordinates of mock satellites, rotated such that the LMC is fixed at the Cartesian coordinates from the previous function

    Args:
    satellite_properties (dict): dict containing properties of satellites
    halo_data (dict): dict containing properties of host and subhalos 
    cosmo_params (dict): dict containing cosmological parameters
    lmc_cartesian_coords (array): Cartesian coordinates of LMC analog
    lmc_ind (int): index of LMC analog in 'Halo_subs'

    Returns:
    satellite_properties (dict): dict containing properties of satellites including rotated positions
    """
    pos = satellite_properties['pos']

    #Change satellite positions to observer frame
    pos_observer_frame = transform_to_observer_frame(pos,observer_ind)

    #Set LMC coordinates
    mock_lmc_coords = (pos_observer_frame[lmc_ind][0], pos_observer_frame[lmc_ind][1], pos_observer_frame[lmc_ind][2])
    lmc_cartesian_coords = get_lmc_coords(halo_data,cosmo_params,lmc_ind)

    #Perform coordinate rotation
    rotated_pos = rotate_about_vector(pos_observer_frame,mock_lmc_coords,lmc_cartesian_coords)
    satellite_properties['rotated_pos'] = rotated_pos

    return satellite_properties