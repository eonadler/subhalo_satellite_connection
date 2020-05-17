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
    mw_sats = mw_sats[mw_sats['type2']>=classification_threshold]

    for survey in surveys:
        sats['{}'.format(survey)] = mw_sats[mw_sats['survey'].astype('str')==survey]

    return sats


def get_true_counts(surveys=['des','ps1']):
    true_counts = {}
    sats = load_satellites()

    for survey in surveys:
        mu = Mr_to_mu(sats['{}'.format(survey)]['m_v'],sats['{}'.format(survey)]['r_physical'])
        bright_mu_split, dim_mu_split = count_Mr_split(sats['{}'.format(survey)]['m_v'],mu,np.ones(len(mu)))
        true_counts['{}_bright_mu_split'.format(survey)] = bright_mu_split
        true_counts['{}_dim_mu_split'.format(survey)] = dim_mu_split

    return true_counts