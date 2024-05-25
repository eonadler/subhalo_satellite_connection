#Set paths
import sys
sys.path.append('../../utils')

#Imports
import numpy as np
import pickle
from chainconsumer import ChainConsumer
import emcee
import progressbar
import matplotlib.pyplot as plt
import time
import data_loader
import data_vector
import posterior_wdm

###

MPI = False

if MPI == True:
    from schwimmbad import MPIPool

else:
    import os
    os.environ["OMP_NUM_THREADS"] = "4"
    from multiprocessing import Pool
    
###

true_params = {}
true_params['alpha'] = -1.436
true_params['mpeak_cut'] = 7.85
true_params['B'] = 0.93
true_params['sigma_M'] = 0.2
true_params['sigma_r'] = 0.63
true_params['A'] = 0.037
true_params['n'] = 1.
true_params['sigma_mpeak'] = 0.2
true_params['Mhm'] = 5.0

hyper_params = {}
hyper_params['gamma_M'] = 0.
hyper_params['xi_8'] = 0.0
hyper_params['xi_9'] = 0.0
hyper_params['xi_10'] = 0.0

vpeak_Mr_interp = data_loader.load_interpolator()

###

#Luminosity binning
bins = np.linspace(-20,0,15)

#All possible host halos
halo_numbers_all = np.array([23,88,119,188,247,268,327,349,364,374,414,415,416,440,460,469,490,530,558,
               567,570,606,628,641,675,718,738,749,797,800,825,829,852,878,881,925,926,
               937,939,967,990,9749,9829])
halo_data_all = data_loader.load_halo_data_all()
nhalos_all = len(halo_numbers_all)

#Number of modeal realizations
n_realizations = 10

#A small number of steps for illustration
n_steps = 10000
ndim, nwalkers = 9, 36

moves = [
    [
        (emcee.moves.StretchMove(), 0.5),
        (emcee.moves.KDEMove(), 0.5)
    ],
]

nhosts = 2

###

for i in [nhosts]:
    halo_numbers = halo_numbers_all[0:i]
    datavector = {}
    global data
    data = data_loader.load_halo_data(halo_numbers,halo_data_all)
    global total_data
    total_data = data_loader.load_halo_data_total(data,halo_numbers)
    ###
    datavector = data_vector.generate_datavector(data,n_realizations,bins,true_params,hyper_params,
                                                halo_numbers,vpeak_Mr_interp)
    with open('datavector_{}.pickle'.format(i),
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
        if MPI == True:
            with MPIPool() as pool:
                np.random.seed(93284)
                sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior_wdm.lnpost_global_data, moves=move, pool=pool)
                start = time.time()
                sampler.run_mcmc(p0, n_steps, progress=True)
                end = time.time()
                print("Autocorrelation time: {0:.2f} steps".format(sampler.get_autocorr_time(quiet=True)[0]))
                multi_time = end - start
                print("Runtime: {0:.1f} seconds".format(multi_time))
        else:
            with Pool(processes=16) as pool:
                np.random.seed(93284)
                sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior_wdm.lnpost_global_data, moves=move, pool=pool,
                                               args=(total_data,datavector,hyper_params,n_realizations,bins,halo_numbers,vpeak_Mr_interp))
                start = time.time()
                sampler.run_mcmc(p0, n_steps, progress=True)
                end = time.time()
                print("Autocorrelation time: {0:.2f} steps".format(sampler.get_autocorr_time(quiet=True)[0]))
                multi_time = end - start
                print("Runtime: {0:.1f} seconds".format(multi_time))
        ###
    samples = sampler.get_chain(flat=True, discard=1000)#.reshape((-1, ndim))
    print(samples[:,5])
    samples[:,2] = samples[:,2] - np.log10(0.7)
    samples[:,5] = 1000.0*samples[:,5]
    np.save('samples_{}'.format(i),samples)
        ###
    c = ChainConsumer()
    c.add_chain(samples, parameters=[r"$\alpha$",r"$\sigma_M$",r"$\mathcal{M}_{50}$",r"$\mathcal{B}$",
                                     r"$\mathcal{S}_{\mathrm{gal}}$",r"$\mathcal{A}$",r"$\sigma_{\log R}$",r"$n$",r"$M_{\mathrm{hm}}$"])
    c.configure(colors=["#04859B", "#003660"], shade=[True,False], shade_alpha=0.5, bar_shade=True,spacing=3.0,
                    diagonal_tick_labels=False, tick_font_size=14, label_font_size=13, sigma2d=False,max_ticks=3, 
                    summary=True,kde=False)
    fig = c.plotter.plot(figsize=(12,12),extents=[[-2,-1.1],[0,2.0],[5,10],[10**-5,3],[0.025,1.0],[10,500],[0,2.0],[0,2],[5,10]],
    truth=[true_params['alpha'], true_params['sigma_M'], true_params['mpeak_cut']-np.log10(0.7), true_params['B'], 
           true_params['sigma_mpeak'], 1000.0*true_params['A'], true_params['sigma_r'], true_params['n'], true_params['Mhm']],
    display=True,filename="corner_{}.pdf".format(i))
