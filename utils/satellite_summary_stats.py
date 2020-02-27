def count_Mr(Mr,bins,prob,Mr_max):
    """
    Args:
        Mr (array of floats): Magnitude of each satellite
        bins (array of floats): Magnitude bins
        prob (array of floats): detection probability of each satellite
        Mr_max (float): dimmest magnitude considered
    
    Returns:
        Average density (per unit magnitude) of suriving halos in each magnitude bin.
    """
    idx = np.isfinite(Mr)
    idx_in_bins = (Mr >= bins[0]) & (Mr <= bins[-1]) & (Mr <= Mr_max)
    norm = np.sum(prob[idx & idx_in_bins])
    return np.histogram(Mr[idx], bins=bins, weights=prob[idx], density=True)[0] * norm

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
        Average density (per unit magnitude) of suriving halos in each magnitude bin.
    """
    idx = np.isfinite(Mr)
    idx_in_bins = (Mr >= bins[0]) & (Mr <= bins[-1]) & (Mr <= Mrmax)
    norm1 = np.sum(prob[np.logical_and(idx,mu<28) & idx_in_bins])
    norm2 = np.sum(prob[np.logical_and(idx,mu>28) & idx_in_bins])
    count_Mr_1 = np.histogram(Mr[np.logical_and(idx,mu<28)], bins=bins, weights=prob[np.logical_and(idx,mu<28)], density=True)[0] * norm1
    count_Mr_2 = np.histogram(Mr[np.logical_and(idx,mu>28)], bins=bins, weights=prob[np.logical_and(idx,mu>28)], density=True)[0] * norm2
    return count_Mr_1, count_Mr_2