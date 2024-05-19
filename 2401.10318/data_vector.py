#Imports
import numpy as np
import data_loader
import model
from scipy.special import erf

def count_Mr_num(Mr,mu,bins,prob):
    """
    Args:
        Mr (array of floats): Magnitude of each subhalo
        mu (array of floats): Surface brightness 
        bins (array of floats): Magnitude bins
        prob (array of floats): disruption probability of each subhalo
    
    Returns:
        Average number of suriving halos in each magnitude bin.
    """
    idx = (np.isfinite(Mr)) & (Mr<0.) & (mu<32.)
    return np.histogram(Mr[np.logical_and(idx,mu < 28)], bins=bins, weights=1.-prob[np.logical_and(idx,mu < 28)])[0], np.histogram(Mr[np.logical_and(idx,mu > 28)], bins=bins, weights=1.-prob[np.logical_and(idx,mu > 28)])[0]

def properties_given_theta(alpha,Halo_subs,rvir,B,Halo_ML_prob,sigma_M,gamma_M,sigma_r,sigma_mpeak,A,n,Mhm,
                           bin_modulation,bin_lower,bin_upper,mpeak_cut,h=0.7):
    """ Returns subhalo magnitude (Mr), half-light radii (r1/2), and disruption probability (prob) 
        given model parameters (and ...) """
    ###
    vpeak_Mr_interp = data_loader.load_interpolator()
    ###
    mpeak_prob = (0.5*(1.+erf((np.log10(Halo_subs['mpeak'])-mpeak_cut)/(np.sqrt(2)*sigma_mpeak))))
    mpeak_prob[np.logical_and(np.log10(Halo_subs['mpeak']/h)>bin_lower,
                              np.log10(Halo_subs['mpeak']/h)<bin_upper)]*=bin_modulation
    Halo_r12 = (A *(rvir/(10.*0.702))**n)
    Halo_ML_prob = 1.-((1.-(Halo_ML_prob)**(1/B))*mpeak_prob*((1.+(2.7*h*(10**Mhm)/Halo_subs['mpeak'])**1.0)**(-0.99)))
    ###
    idx = np.argsort(np.argsort(Halo_subs['vpeak']))
    sigma_M_eff = sigma_M + -1.*gamma_M*(np.log10(Halo_subs['mpeak']/h)-10.)
    Halo_dmo_Mr_mean = vpeak_Mr_interp(Halo_subs['vpeak'], alpha)[idx]
    Halo_L_mean = 10**((-1.*Halo_dmo_Mr_mean + 4.81)/2.5 + np.log10(2))
    Halo_L = np.random.lognormal(np.log(Halo_L_mean),(np.log(10)*sigma_M_eff.clip(min=0.001)))
    Halo_dmo_Mr = -1.*(2.5*(np.log10(Halo_L)-np.log10(2))-4.81)
    Halo_r12 = np.random.lognormal(np.log(Halo_r12),np.log(10)*sigma_r)#.clip(min=0.02)
    ###
    return Halo_dmo_Mr, Halo_r12, Halo_ML_prob

def generate_datavector(data,n_realizations_datavector,bins,true_params,hyper_params,halo_numbers):
    datavector = {}
    for j,d in enumerate(data):
        print(halo_numbers[j])
        datavector[halo_numbers[j]] = []
        count1_all = []
        count2_all = []
        ###
        for k in range(n_realizations_datavector):
            Mr, r12, probs = properties_given_theta(true_params['alpha'], d['Halo_subs'], d['rvir'], true_params['B'],
                                                    d['Halo_ML_prob'], true_params['sigma_M'], hyper_params['gamma_M'],
                                                    true_params['sigma_r'], true_params['sigma_mpeak'], true_params['A'], 
                                                    true_params['n'],hyper_params['Mhm'], hyper_params['bin_modulation'], 
                                                    hyper_params['bin_lower'],hyper_params['bin_upper'],true_params['mpeak_cut'])
            mu = model.Mr_to_mu(Mr,r12)
            count1, count2 = count_Mr_num(Mr, mu, bins, probs)
            count1_all.append(count1)
            count2_all.append(count2)
        ###
        datavector[halo_numbers[j]].append(np.mean(count1_all,axis=0))
        datavector[halo_numbers[j]].append(np.mean(count2_all,axis=0))
        datavector[halo_numbers[j]] = np.asarray(datavector[halo_numbers[j]])
    ###
    datavector['all'] = np.asarray([datavector[halo_numbers[i]] for i in range(len(data))])
    return datavector