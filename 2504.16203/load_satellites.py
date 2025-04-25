#Set paths
import sys
sys.path.append('../1912.03303')
sys.path.append('../utils')

#Imports
import numpy as np
from predict_satellites import Mr_to_mu
from satellite_summary_stats import count_Mr_split

def load_satellites(surveys=['des','ps1'],classification_threshold=3):
    sats = {}
    mw_sats = np.recfromcsv('../Data/mw_sats_master.csv')
    #mw_sats = mw_sats[mw_sats['type2']>=classification_threshold]
    confirmed_sats = {'des': ['Fornax', 'Sculptor', 'Reticulum II', 'Tucana II', 'Grus II',
                             'Horologium I', 'Tucana III', 'Tucana IV', 'Phoenix II', 'Horologium II',
                             'Tucana V', 'Pictor I', 'Columba I', 'Cetus II', 'Grus I', 'Reticulum III'],
                        'ps1': ['Leo I', 'Leo II', 
                                'Draco', 'Ursa Minor', 'Sextans', 'Canes Venatici I', 'Bootes I', 'Ursa Major II',
                                'Coma Berenices', 'Sagittarius II', 'Willman 1', 'Canes Venatici II', 'Segue 1', 'Segue 2', 
                                'Crater II', 'Draco II', 'Triangulum II', 'Hercules']}

    for survey in surveys:
        sats['{}'.format(survey)] = []
        for sat in confirmed_sats[survey]:
            sats['{}'.format(survey)].append(mw_sats[mw_sats['name']==str.encode(sat)])
        sats['{}'.format(survey)] = np.array(sats['{}'.format(survey)])

    return sats


def get_true_counts(surveys=['des','ps1']):
    true_counts = {}
    sats = load_satellites()

    for survey in surveys:
        mu = Mr_to_mu(np.ravel(sats['{}'.format(survey)]['m_v']),np.ravel(sats['{}'.format(survey)]['r_physical']))
        bright_mu_split, dim_mu_split = count_Mr_split(np.ravel(sats['{}'.format(survey)]['m_v']),mu,np.ones(len(mu)))
        true_counts['{}_bright_mu_split'.format(survey)] = bright_mu_split
        true_counts['{}_dim_mu_split'.format(survey)] = dim_mu_split

    return true_counts