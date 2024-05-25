#Imports
import numpy as np
import data_loader
import model
from scipy.special import erf

def generate_datavector(data,n_realizations_datavector,bins,true_params,hyper_params,halo_numbers,vpeak_Mr_interp):
    datavector = {}
    for j,d in enumerate(data):
        print(halo_numbers[j])
        datavector[halo_numbers[j]] = []
        count1_all = []
        count2_all = []
        ###
        for k in range(n_realizations_datavector):
            Mr, r12, probs = model.properties_given_theta(true_params['alpha'], d['Halo_subs'], d['rvir'], true_params['B'],
                                                    d['Halo_ML_prob'], true_params['sigma_M'], hyper_params['gamma_M'],
                                                    true_params['sigma_r'], true_params['sigma_mpeak'], true_params['A'], 
                                                    true_params['n'],true_params['Mhm'], true_params['mpeak_cut'],
                                                    hyper_params['xi_8'],hyper_params['xi_9'],hyper_params['xi_10'],vpeak_Mr_interp)
            mu = model.Mr_to_mu(Mr,r12)
            count1, count2 = model.count_Mr_num(Mr, mu, bins, probs)
            count1_all.append(count1)
            count2_all.append(count2)
        ###
        datavector[halo_numbers[j]].append(np.mean(count1_all,axis=0))
        datavector[halo_numbers[j]].append(np.mean(count2_all,axis=0))
        datavector[halo_numbers[j]] = np.asarray(datavector[halo_numbers[j]])
    ###
    datavector['all'] = np.asarray([datavector[halo_numbers[i]] for i in range(len(data))])
    return datavector