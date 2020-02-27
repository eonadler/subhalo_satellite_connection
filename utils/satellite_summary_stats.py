def count_Mr(Mr,bins,prob,Mr_max):
    """
    Args:
        Mr (array of floats): Magnitude of each satellite
        bins (array of floats): Magnitude bins
        prob (array of floats): detection probability of each satellite
        Mr_max (float): dimmest magnitude considered
    
    Returns:
        Average density (per unit magnitude) of surviving satellites in each magnitude bin.
    """
    idx = np.isfinite(Mr)
    idx_in_bins = (Mr >= bins[0]) & (Mr <= bins[-1]) & (Mr <= Mr_max)
    norm = np.sum(prob[idx & idx_in_bins])
    count_Mr = np.histogram(Mr[idx], bins=bins, weights=prob[idx], density=True)[0] * norm
    count_Mr_num = np.histogram(Mr[idx], bins=bins, weights=prob[idx])[0]
    return count_Mr, count_Mr_num

def count_Mr_split(Mr,mu,bins,prob,Mr_max,mu_split=28.):
    """
    Args:
        Mr (array of floats): Magnitude of each satellite
        mu (array of floats): central surface brightness of each satellite
        bins (array of floats): Magnitude bins
        prob (array of floats): detection probability of each satellite
        Mr_max (float): dimmest satellite considered
        mu_split: surface brightness at which satellite sample is split 
    
    Returns:
        Average density (per unit magnitude) and absolute number of surviving satellites in each magnitude bin.
    """
    idx = np.isfinite(Mr)
    idx_in_bins = (Mr >= bins[0]) & (Mr <= bins[-1]) & (Mr <= Mrmax)
    norm1 = np.sum(prob[np.logical_and(idx,mu<mu_split) & idx_in_bins])
    norm2 = np.sum(prob[np.logical_and(idx,mu>mu_split) & idx_in_bins])
    #Normed distributions
    count_Mr_1 = np.histogram(Mr[np.logical_and(idx,mu<mu_split)], bins=bins, weights=prob[np.logical_and(idx,mu<mu_split)], density=True)[0] * norm1
    count_Mr_2 = np.histogram(Mr[np.logical_and(idx,mu>mu_split)], bins=bins, weights=prob[np.logical_and(idx,mu>mu_split)], density=True)[0] * norm2
    #Number counts
    count_Mr_num_1 = np.histogram(Mr[np.logical_and(idx,mu < mu_split)], bins=bins, weights=prob[np.logical_and(idx,mu < mu_split)])[0]
    count_Mr_num_2 = np.histogram(Mr[np.logical_and(idx,mu > mu_split)], bins=bins, weights=prob[np.logical_and(idx,mu > mu_split)])[0]
    return count_Mr_1, count_Mr_2, count_Mr_num_1, count_Mr_num_2

def N_gr_Mr_new(Mr,radii,prob,bins,radius_max,Mr_min):
    """
    Returns a luminosity function of surviving satellite, subject to maxmimum
    distance and brightest magnitude.
    
    Args:
        Mr (array of floats): Magnitude of each satellite
        radii (array of floats): Galactocentric radius of each satellite
        prob (array of floats): Disruption probability of each satellite
        bins (array of floats): Magnitude bin edges
        radius_max (float): Maximum radius within which to consider satellite
        Mr_min (float): brightest magnitude considered
    
    Returns:
        Cumulative number of surviving satellites less than each magnitude
    """
    bin_edges = np.hstack([-np.inf, bins])
    idx = (radii < radius_max) & (Mr > Mr_min) & np.isfinite(Mr)
    n = np.histogram(Mr[idx], bins=bin_edges, weights=prob[idx])[0]
    return np.cumsum(n)

def N_less_r(Mr,radii,prob,binsr,Mr_min):
    """
    Calculates the cumulative distribution function of Galactocentric radius, subject
    to some cuts on magnitude.
    
    Inputs:
        Mr (array): magnitude of each satellite
        radii (array): Galactocentric radius of each satellite
        prob (array): detection probability of each satellite
        binsr (array): Galactocentric radius bin edges
        Mr_min (float): brightest magnitude considered
    
    Returns:
        Array with length `binsr.size-1`, with number of satellite at half-light radii less than
        distances `binsr[1:]`.
    """
    idx = (Mr > Mr_min) & np.isfinite(Mr)
    radii = radii[idx]
    prob = prob[idx]
    n_in_bin = np.histogram(radii, bins=binsr, weights=prob)[0]
    return np.cumsum(n_in_bin)

def N_less_r12(Mr,radii,r12,prob,binsr12,radius_max,Mr_min):
    """
    Calculates the cumulative distribution function of satellite sizes, subject
    to cuts on magnitude and Galactocentric distance.
    
    Inputs:
        Mr (array): magnitude of each satellite
        radii (array): Galactocentric radius of each satellite
        prob (array): detection probability of each satellite
        binsr12 (array): half-light radius bin edges
        radius_max (float): Galactocentric radius within which satellites are counted
        Mr_min (float): brightest magnitude considered
    
    Returns:
        Array with length `binsr12.size-1`, with number of satellite of half-light radii less than `binsr12[1:]`.
    """
    idx = (Mr > Mr_min) & np.isfinite(Mr) & (radii < radius_max)
    r12 = r12[idx]
    prob = prob[idx]
    n_in_bin = np.histogram(r12, bins=binsr12, weights=prob)[0]
    return np.cumsum(n_in_bin)