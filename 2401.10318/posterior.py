#Imports
import numpy as np
import model
from stats import marginalized_like_vectorized

#Standard galaxy--halo connection priors
def log_prior(alpha, sigma_M, mpeak_cut, B, sigma_mpeak, A, sigma_r, n):
    """ Returns the log prior of the data """
    alpha_min,alpha_max = -2,-1.1
    B_min,B_max = 10**-5,3.0
    mpeak_cut_min, mpeak_cut_max = 4.85, 9.85
    sigma_M_min, sigma_M_max = 0.0,2.0
    sigma_mpeak_min, sigma_mpeak_max = 0.025, 1.0
    A_min, A_max = 0.01, 0.5
    sigma_r_min, sigma_r_max = 0.0, 2.0
    n_min, n_max = 0.0, 2.0
    if (alpha > alpha_min)*(alpha < alpha_max)*(B > B_min)*(B < B_max)*(mpeak_cut > mpeak_cut_min)*(mpeak_cut < mpeak_cut_max)*(sigma_M > sigma_M_min)*(sigma_M < sigma_M_max)*(sigma_mpeak > sigma_mpeak_min)*(sigma_mpeak < sigma_mpeak_max)*(A > A_min)*(A<A_max)*(sigma_r>sigma_r_min)*(sigma_r<sigma_r_max)*(n>n_min)*(n<n_max):
        value = np.log(1.0/(1.0+(alpha**2))) - ((n-1)**2)/(2*(0.5**2)) - ((np.log(B))**2)/(2*(0.5**2))
    else:
        value = -np.inf
    return value

#Tight galaxy--halo connection priors
def log_prior_tight(alpha, sigma_M, mpeak_cut, B, sigma_mpeak, A, sigma_r, n):
    """ Returns the log prior of the data """
    alpha_min,alpha_max = -1.5,-1.3#-2,-1.1
    B_min,B_max = 0.5,1.5#10**-5,3.0
    mpeak_cut_min, mpeak_cut_max = 4.85, 9.85
    sigma_M_min, sigma_M_max = 0.0,0.4#0.0,2.0
    sigma_mpeak_min, sigma_mpeak_max = 0.025, 1.0
    A_min, A_max = 0.01, 0.5
    sigma_r_min, sigma_r_max = 0.0, 2.0
    n_min, n_max = 0.0, 2.0
    if (alpha > alpha_min)*(alpha < alpha_max)*(B > B_min)*(B < B_max)*(mpeak_cut > mpeak_cut_min)*(mpeak_cut < mpeak_cut_max)*(sigma_M > sigma_M_min)*(sigma_M < sigma_M_max)*(sigma_mpeak > sigma_mpeak_min)*(sigma_mpeak < sigma_mpeak_max)*(A > A_min)*(A<A_max)*(sigma_r>sigma_r_min)*(sigma_r<sigma_r_max)*(n>n_min)*(n<n_max):
        value = np.log(1.0/(1.0+(alpha**2))) - ((n-1)**2)/(2*(0.5**2)) - ((np.log(B))**2)/(2*(0.5**2))
    else:
        value = -np.inf
    return value

def lnpost_global_data_v3(y,total_data,datavector,hyper_params,n_realizations,bins,halo_numbers,tight_prior=False):
    """ Returns the log posterior of the data """
    if tight_prior == True:
        log_prior_y = log_prior_tight(*y)
    else:
        log_prior_y = log_prior(*y)
    if np.isinf(log_prior_y):
        return log_prior_y
    elif np.isnan(log_prior_y):
        return -np.inf
    ###
    Mr, r12, probs = model.properties_given_theta_v3(y[0], total_data['Halo_subs'], total_data['rvir'], y[3],
                                                     total_data['Halo_ML_prob'], y[1], hyper_params['gamma_M'],
                                                     y[6],y[4],y[5],y[7],hyper_params['Mhm'],hyper_params['bin_modulation'],
                                                     hyper_params['bin_lower'],hyper_params['bin_upper'],n_realizations,y[2])
    mu = model.Mr_to_mu(Mr,r12)
    splits = total_data['splits']
    splits = [0] + splits
    count_arr = []
    for i in range(n_realizations):
        for j in range(len(splits)-1):
            count = model.count_Mr_num_v2(Mr[i, splits[j]:splits[j+1]], mu[i, splits[j]:splits[j+1]], 
                                          bins, probs[i, splits[j]:splits[j+1]])
            count_arr.append(count)
    count_arr = np.asarray(count_arr)
    count_arr = count_arr.reshape(n_realizations,len(halo_numbers),2,len(bins)-1)
    all_like_combined = np.sum(marginalized_like_vectorized(datavector['all'], count_arr))
    if np.isnan(all_like_combined):
        return -np.inf
    else:
        return all_like_combined + log_prior_y