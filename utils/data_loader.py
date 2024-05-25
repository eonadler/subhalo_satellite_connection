#Imports
import numpy as np
import pickle

#Abundance matching
def load_interpolator():
    with open('../../Data/interpolator.pkl', 'rb') as ff:
        vpeak_Mr_interp = pickle.load(ff,encoding='latin1')
    return vpeak_Mr_interp

#All halo dictionary
def load_halo_data_all():
    with open('../../Data/halo_data_all.pkl', 'rb') as ff:
        halo_data_all = pickle.load(ff,encoding='latin1')
    return halo_data_all

#Load halo data
def load_halo_data(halo_numbers,halo_data_all):
    data = []
    ###
    for k in halo_numbers:
        data.append(halo_data_all[k])
    return data

#Load segmented halo data
def load_halo_data_total(data,halo_numbers):
    d = data[0]
    total = {key: np.concatenate([d[key] for d in data], axis=None) for key in d.keys()}
    total['splits'] = [len(data[i]['Halo_subs']) for i in range(len(halo_numbers))]    
    return total
