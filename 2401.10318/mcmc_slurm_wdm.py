#Imports
import os
#os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import yaml
import os
import pickle
import scipy
from pylab import figure, axes, pie, title, show
from chainconsumer import ChainConsumer
import corner
import emcee
from AbundanceMatching import *
from scipy.special import loggamma, erf
import astropy.units as u
from numba import jit
import progressbar
from stats import marginalized_like_vectorized

from numpy.random import default_rng
global rng
rng = default_rng()

from multiprocessing import Pool
from schwimmbad import MPIPool
import time

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("ticks")
green = sns.light_palette("green")[5]
cyan = sns.light_palette("turquoise")[5]
blue = sns.light_palette("blue")[5]
red = sns.light_palette("red")[5]
pink = sns.light_palette("magenta")[5]

###

#Abundance matching
with open('../Data/interpolator.pkl', 'rb') as ff:
    vpeak_Mr_interp = pickle.load(ff,encoding='latin1')

#Luminosity binning
classical_bins = np.linspace(-20,0,15)

#Cosmological parameters
omega_b = 0#0.047
omega_m = 0.286
h = 0.7
gamma_M  = 0.

#Host halos to use
halo_numbers_all = np.array([23,88,119,188,247,268,327,349,364,374,414,415,416,440,460,469,490,530,558,
               567,570,606,628,641,675,718,738,749,797,800,825,829,852,878,881,925,926,
               937,939,967,990,9749,9829])
nhalos_all = len(halo_numbers_all)

#Number of model realizations
n_realizations = 10

def mwdm(Mhm):
    """ Converts a given half mode mass to the corresponding warm dark matter mass """
    lambda_hm = 2.*(((3./4)*Mhm/(np.pi*rho_crit))**(1./3))
    lambda_fs = lambda_hm/13.93
    mwdm = (h*lambda_fs/(0.049*((omega_m/0.25)**(0.11))*((h/0.7)**1.22)))**(1./-1.11)
    return mwdm

#All halo dictionary
def load_halo_data_all():
    with open('../Data/halo_data_all.pkl', 'rb') as ff:
        halo_data_all = pickle.load(ff,encoding='latin1')
    return halo_data_all

#Load halo data
def load_halo_data(halo_numbers,halo_data_all):
    data = []
    ###
    for k in halo_numbers:
        data.append(halo_data_all[k])
    return data

def load_halo_data_total(data):
    d = data[0]
    total = {key: np.concatenate([d[key] for d in data], axis=None) for key in d.keys()}
    total['splits'] = [len(data[i]['Halo_subs']) for i in range(len(halo_numbers))]    
    return total

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
    idx = np.isfinite(Mr)
    return np.histogram(Mr[np.logical_and(idx,mu < 28)], bins=bins, weights=1.-prob[np.logical_and(idx,mu < 28)])[0], np.histogram(Mr[np.logical_and(idx,mu > 28)], bins=bins, weights=1.-prob[np.logical_and(idx,mu > 28)])[0]

def properties_given_theta(alpha,Halo_subs,rvir,B,Halo_ML_prob,sigma_M,gamma_M,sigma_r,sigma_mpeak,A,n,Mhm,
                           bin_modulation,bin_lower,bin_upper,mpeak_cut):
    """ Returns subhalo magnitude (Mr), half-light radii (r1/2), and disruption probability (prob) 
        given model parameters (and ...) """
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


def log_prior(alpha, sigma_M, mpeak_cut, B, sigma_mpeak, A, sigma_r, n, Mhm):
    """ Returns the log prior of the data """
    alpha_min,alpha_max = -2,-1.1
    B_min,B_max = 10**-5,3.0
    mpeak_cut_min, mpeak_cut_max = 4.85, 9.85
    sigma_M_min, sigma_M_max = 0.0,2.0
    sigma_mpeak_min, sigma_mpeak_max = 0.025, 1.0
    A_min, A_max = 0.01, 0.5
    sigma_r_min, sigma_r_max = 0.0, 2.0
    n_min, n_max = 0.0, 2.0
    Mhm_min, Mhm_max = 5.0, 10.0
    if (alpha > alpha_min)*(alpha < alpha_max)*(B > B_min)*(B < B_max)*(mpeak_cut > mpeak_cut_min)*(mpeak_cut < mpeak_cut_max)*(sigma_M > sigma_M_min)*(sigma_M < sigma_M_max)*(sigma_mpeak > sigma_mpeak_min)*(sigma_mpeak < sigma_mpeak_max)*(A > A_min)*(A<A_max)*(sigma_r>sigma_r_min)*(sigma_r<sigma_r_max)*(n>n_min)*(n<n_max)*(Mhm>Mhm_min)*(Mhm<Mhm_max):
        value = np.log(1.0/(1.0+(alpha**2))) - ((n-1)**2)/(2*(0.5**2)) - ((np.log(B))**2)/(2*(0.5**2))
    else:
        value = -np.inf
    return value

def properties_given_theta_v3(alpha,Halo_subs,rvir,B,Halo_ML_prob,sigma_M,gamma_M,sigma_r,sigma_mpeak,A,n,Mhm,
                           bin_modulation,bin_lower,bin_upper, n_realizations,mpeak_cut):
    """ Returns subhalo magnitude (Mr), half-light radii (r1/2), and disruption probability (prob) 
        given model parameters (and ...) """
    ###
    mpeak = Halo_subs['mpeak'] 
    vpeak = Halo_subs['vpeak']
    log_mpeak = np.log10(mpeak)
    log_h = np.log10(h)
    mpeak_prob = (0.5*(1.+erf((log_mpeak-mpeak_cut)/(np.sqrt(2)*sigma_mpeak))))
    mpeak_prob[np.logical_and((log_mpeak-log_h)>bin_lower,
                              (log_mpeak-log_h)<bin_upper)]*=bin_modulation
    Halo_r12 = (A *(rvir/(10.*0.702))**n)
    nh = len(Halo_r12)
    Halo_ML_prob = 1.-((1.-(Halo_ML_prob)**(1/B))*mpeak_prob*((1.+(2.7*h*(10**Mhm)/mpeak)**1.0)**(-0.99)))
    Halo_ML_prob = np.tile(Halo_ML_prob, (n_realizations,1))
    ###
    idx = np.argsort(np.argsort(vpeak))
    sigma_M_eff = sigma_M + -1.*gamma_M*((log_mpeak-log_h)-10.)
    Halo_dmo_Mr_mean = vpeak_Mr_interp(vpeak, alpha)[idx]
    Halo_L_mean = 10**((-1.*Halo_dmo_Mr_mean + 4.81)/2.5 + np.log10(2))
    Halo_L = rng.lognormal(
        np.log(Halo_L_mean),
        np.log(10)*sigma_M_eff.clip(min=0.001), 
        size=(n_realizations, nh)
    )
    Halo_dmo_Mr = -1.*(2.5*(np.log10(Halo_L)-np.log10(2))-4.81)
    Halo_r12 = rng.lognormal(
        np.log(Halo_r12),
        np.log(10)*sigma_r, 
        size=(n_realizations, nh)
    )
    ###
    return Halo_dmo_Mr, Halo_r12, Halo_ML_prob

def count_Mr_num_v2(Mr,mu,bins,prob):
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
    return (
    np.histogram(Mr[lidx], bins=bins, weights=1.-prob[lidx])[0], 
    np.histogram(Mr[ridx], bins=bins, weights=1.-prob[ridx])[0]
    )


def lnpost_global_data_v3(y):
    """ Returns the log posterior of the data """
    log_prior_y = log_prior(*y)
    if np.isinf(log_prior_y):
        return log_prior_y
    elif np.isnan(log_prior_y):
        return -np.inf
    
    Mr, r12, probs = properties_given_theta_v3(
        y[0], total_data['Halo_subs'], total_data['rvir'], y[3],
        total_data['Halo_ML_prob'], y[1], gamma_M, y[6],y[4],y[5],y[7],y[8],
        1.,9.5,10., n_realizations,y[2]) 
    mu = Mr+36.57+2.5*np.log10(2*np.pi*((r12)**2))
    splits = total_data['splits']
    splits = [0] + splits
    count_arr = []
    for i in range(n_realizations):
        for j in range(len(splits)-1):
            count = count_Mr_num_v2(
            Mr[i, splits[j]:splits[j+1]], 
            mu[i, splits[j]:splits[j+1]], 
            classical_bins, 
            probs[i, splits[j]:splits[j+1]]
        )
            count_arr.append(count)
    count_arr = np.asarray(count_arr)
    count_arr = count_arr.reshape(n_realizations,len(halo_numbers),2,len(classical_bins)-1)
    all_like_combined = np.sum(marginalized_like_vectorized(datavector['all'], count_arr))
    if np.isnan(all_like_combined):
        return -np.inf
    else:
        return all_like_combined + log_prior_y

ndim, nwalkers = 9, 32

p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
p0[:,0] = -1.3
p0[:,1] = 0.5
p0[:,2] = 8
p0[:,3] = 0.5
p0[:,4] = 0.5
p0[:,5] = 0.02
p0[:,6] = 0.1
p0[:,7] = 0.5
p0[:,8] = 7.0
p0 = p0 + 0.1*np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

moves = [
    [
        (emcee.moves.StretchMove(), 0.5),
        (emcee.moves.KDEMove(), 0.5)
    ],
]

###

n_steps = 100000
nwalkers = 450

for i in [2]:
    halo_numbers = halo_numbers_all[0:i]
    datavector = {}
    global data
    data = load_halo_data(halo_numbers)
    global total_data
    total_data = load_halo_data_total(data)
    ###
    for j,d in enumerate(data):
        datavector[halo_numbers[j]] = []
        alpha, mpeak_cut, B, sigma_M, sigma_r, A, n, sigma_mpeak, Mhm = -1.436, 7.85, 0.93, 0.2, 0.63, 0.037, 1., 0.2, 5.0
        count1_all = []
        count2_all = []
        for k in range(n_realizations):
            Mr, r12, probs = properties_given_theta(
                    alpha, d['Halo_subs'], d['rvir'], B, d['Halo_ML_prob'], sigma_M, gamma_M, sigma_r, sigma_mpeak,A,n,Mhm,
                1.,7.,1.,mpeak_cut)
            mu = Mr+36.57+2.5*np.log10(2*np.pi*((r12)**2))
            ind = np.logical_and(Mr<0.,mu<32.)
            count1, count2 = count_Mr_num(Mr[ind], mu[ind], classical_bins, probs[ind])
            count1_all.append(count1)
            count2_all.append(count2)
        datavector[halo_numbers[j]].append(np.mean(count1_all,axis=0))
        datavector[halo_numbers[j]].append(np.mean(count2_all,axis=0))
        datavector[halo_numbers[j]] = np.asarray(datavector[halo_numbers[j]])

    datavector['all'] = np.asarray([datavector[halo_numbers[i]] for i in range(len(data))])

    with open('datavector_wdm{}.pickle'.format(i),
              'wb') as handle:
        pickle.dump(datavector, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ###
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    p0[:,0] = -1.4
    p0[:,1] = 0.2
    p0[:,2] = 8
    p0[:,3] = 1
    p0[:,4] = 0.3
    p0[:,5] = 0.03
    p0[:,6] = 0.5
    p0[:,7] = 1
    p0[:,8] = 7.0
    p0 = p0 + 0.1*np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        ###
    for move in moves:
        with MPIPool() as pool:
            print(move)
            np.random.seed(93284)
            sampler = emcee.EnsembleSampler(
                nwalkers, 
                ndim, 
                lnpost_global_data_v3, 
                moves = move,
                pool = pool
            )

            start = time.time()
            sampler.run_mcmc(p0, n_steps, progress=True)
            end = time.time()
            print(
                "Autocorrelation time: {0:.2f} steps".format(
                    sampler.get_autocorr_time(quiet=True)[0]
                )
            )
            multi_time = end - start
            print("Runtime: {0:.1f} seconds".format(multi_time))
        ###
    samples = sampler.get_chain(flat=True, discard=10000)#.reshape((-1, ndim))
    print(samples[:,5])
    samples[:,2] = samples[:,2] - np.log10(0.7)
    samples[:,5] = 1000.0*samples[:,5]
    np.save('samples_wdm{}'.format(i),samples)
        ###
    c = ChainConsumer()
    c.add_chain(samples, parameters=[r"$\alpha$",r"$\sigma_M$",r"$\mathcal{M}_{50}$",r"$\mathcal{B}$",r"$\mathcal{S}_{\mathrm{gal}}$",
                                        r"$\mathcal{A}$",r"$\sigma_{\log R}$",r"$n$",r"$M_{\mathrm{hm}}$"])

    c.configure(colors=["#04859B", "#003660"], shade=[True,False], shade_alpha=0.5, bar_shade=True,spacing=3.0,
                    diagonal_tick_labels=False, tick_font_size=14, label_font_size=13, sigma2d=False,max_ticks=3, 
                    summary=True,kde=False)
    fig = c.plotter.plot(figsize=(12,12),extents=[[-2,-1.1],[0,2.0],[5,10],[10**-5,3],[0.025,1.0],[10,500],[0,2.0],[0,2],[5,10]],
    truth=[-1.436, 0.2, 8, 0.93, 0.2, 1000.0*0.037, 0.63, 1.,5.],
    display=True,filename="corner_wdm{}.pdf".format(i))
