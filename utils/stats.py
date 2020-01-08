def marginalized_poisson_likelihood(k, m):
    """
    Returns Poisson likelihood in a single parameter space bin, marginalized over all rate parameters in that bin, assuming a flat prior on the rate lambda
    (see https://arxiv.org/abs/1809.05542 Eq. 13).

    Args:
    k (int): number of observed events in parameter space bin
    m (vector of floats): vector of observed events in parameter space bin for multiple realizations of model

    Returns:
    marginalized_like (float): likelihood of observing k events given vector m of event realizations in single parameter space bin
    """
    sum_m = np.sum(m)
    marginalized_like = -1.*(1. + sum_m)*np.log((1. + len(m))/len(m)) - k*np.log(1. + len(m)) + loggamma(sum_m + k + 1.) - loggamma(sum_m + 1.) - loggamma(1. + k)
    return marginalized_like

def ln_marginalized_poisson_likelihood(count_binned, rate_binned, bins):
    """
    Returns log of Poisson likelihood over all parameter space bins, marginalized over all rate parameters in each bin, assuming a flat prior on the rate lambda
    (see https://arxiv.org/abs/1809.05542 Eq. 15).

    Args:
    count_binned (vector of ints): vector of number of observed events in parameter space bins
    rate_binned (vector of vector of floats): vector of vector of observed events in parameter space bins, for multiple realizations of model in each bin
    bins (vector of floats): edges of parameter space bins

    Returns:
    ln_marginalized_like (float): likelihood of observing set of events given set of observations in parameter space bins
    """
    ln_marginalized_like = 0.
    for i in range(0,len(bins)-1):
        k = count_binned[i]
        m = rate_binned[:,i]
        ln_marginalized_like += marginalized_poisson_likelihood(k, m)
    return ln_marginalized_like