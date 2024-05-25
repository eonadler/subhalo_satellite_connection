#Imports
import numpy as np
import pickle
import data_loader
from numpy.random import default_rng
from scipy.special import erf

#rng
global rng
rng = default_rng()

def Mr_to_mu(Mr,r12):
    mu = Mr+36.57+2.5*np.log10(2*np.pi*((r12)**2))
    return mu

def properties_given_theta(alpha,Halo_subs,rvir,B,Halo_ML_prob,sigma_M,gamma_M,sigma_r,sigma_mpeak,A,n,Mhm,mpeak_cut,xi_8,xi_9,xi_10,vpeak_Mr_interp,h=0.7):
    """ Returns subhalo magnitude (Mr), half-light radii (r1/2), and disruption probability (prob) 
        given model parameters, for a single host/realization"""
    ###
    mpeak_prob = (0.5*(1.+erf((np.log10(Halo_subs['mpeak'])-mpeak_cut)/(np.sqrt(2)*sigma_mpeak))))
    mpeak_prob[np.logical_and(np.log10(Halo_subs['mpeak']/h)>7.5,np.log10(Halo_subs['mpeak']/h)<8.5)]*=10.**xi_8
    mpeak_prob[np.logical_and(np.log10(Halo_subs['mpeak']/h)>8.5,np.log10(Halo_subs['mpeak']/h)<9.5)]*=10.**xi_9
    mpeak_prob[np.logical_and(np.log10(Halo_subs['mpeak']/h)>9.5,np.log10(Halo_subs['mpeak']/h)<10.5)]*=10.**xi_10
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


def properties_given_theta_multiple(alpha,Halo_subs,rvir,B,Halo_ML_prob,sigma_M,gamma_M,sigma_r,sigma_mpeak,A,n,Mhm,
                           n_realizations,mpeak_cut,xi_8,xi_9,xi_10,vpeak_Mr_interp,h=0.7):
    """ Returns subhalo magnitude (Mr), half-light radii (r1/2), and disruption probability (prob) 
        given model parameters, for multiple hosts and/or realizations"""
    ###
    vpeak_Mr_interp = data_loader.load_interpolator()
    ###
    mpeak = Halo_subs['mpeak'] 
    vpeak = Halo_subs['vpeak']
    log_mpeak = np.log10(mpeak)
    log_h = np.log10(h)
    mpeak_prob = (0.5*(1.+erf((log_mpeak-mpeak_cut)/(np.sqrt(2)*sigma_mpeak))))
    mpeak_prob[np.logical_and(log_mpeak-log_h>7.5,log_mpeak-log_h<8.5)]*=10.**xi_8
    mpeak_prob[np.logical_and(log_mpeak-log_h>8.5,log_mpeak-log_h<9.5)]*=10.**xi_9
    mpeak_prob[np.logical_and(log_mpeak-log_h>9.5,log_mpeak-log_h<10.5)]*=10.**xi_10
    Halo_r12 = (A *(rvir/(10.*0.702))**n)
    nh = len(Halo_r12)
    Halo_ML_prob = 1.-((1.-(Halo_ML_prob)**(1/B))*mpeak_prob*((1.+(2.7*h*(10**Mhm)/mpeak)**1.0)**(-0.99)))
    Halo_ML_prob = np.tile(Halo_ML_prob, (n_realizations,1))
    ###
    idx = np.argsort(np.argsort(vpeak))
    sigma_M_eff = sigma_M + -1.*gamma_M*((log_mpeak-log_h)-10.)
    Halo_dmo_Mr_mean = vpeak_Mr_interp(vpeak, alpha)[idx]
    Halo_L_mean = 10**((-1.*Halo_dmo_Mr_mean + 4.81)/2.5 + np.log10(2))
    Halo_L = rng.lognormal(np.log(Halo_L_mean),np.log(10)*sigma_M_eff.clip(min=0.001), size=(n_realizations, nh))
    Halo_dmo_Mr = -1.*(2.5*(np.log10(Halo_L)-np.log10(2))-4.81)
    Halo_r12 = rng.lognormal(np.log(Halo_r12),np.log(10)*sigma_r,size=(n_realizations, nh))
    ###
    return Halo_dmo_Mr, Halo_r12, Halo_ML_prob

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
    # nr, _ = Mr.shape
    idx = np.logical_and(np.isfinite(Mr), Mr<0.0)
    idx = np.logical_and(idx, mu < 32.0)
    lidx = np.logical_and(idx, mu < 28)
    ridx = np.logical_and(idx, mu > 28)
    ###
    return (np.histogram(Mr[lidx], bins=bins, weights=1.-prob[lidx])[0], 
            np.histogram(Mr[ridx], bins=bins, weights=1.-prob[ridx])[0])