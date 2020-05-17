import yaml
import os
import numpy as np
from numpy import linalg as LA
import healpy as hp
import xgboost as xgb


class surveySelectionFunction:
    
    def __init__(self,config_file):
        self.config = yaml.load(open(config_file))
        self.algorithm = self.config['operation']['algorithm']
        self.classifier = None
        self.loadClassifier()
        
    def load_map(self):
        self.map = hp.read_map(self.config['infile']['stellar_density'])

    def loadClassifier(self):
        if os.path.exists(self.config[self.algorithm]['classifier'] + '.gz') and not os.path.exists(self.config[self.algorithm]['classifier']):
            os.system('gunzip -k %s.gz'%(self.config[self.algorithm]['classifier']))
        reader = open(self.config[self.algorithm]['classifier'])
        self.classifier = xgb.XGBClassifier({'nthread': 4})
        self.classifier.load_model(self.config[self.algorithm]['classifier'])
        
    def predict(self,**kwargs):
        assert self.classifier is not None, 'ERROR'
        ra  = kwargs.pop('ra',None)
        dec = kwargs.pop('dec',None)
        nside = hp.get_nside(self.map)
        kwargs['density'] = self.map[hp.ang2pix(nside,ra,dec,lonlat=True)]
        x_test = []
        for key, operation in self.config['operation']['params_intrinsic']:
            assert operation.lower() in ['linear', 'log'], 'ERROR'
            if operation.lower() == 'linear':
                x_test.append(kwargs[key])
            else:
                x_test.append(np.log10(kwargs[key]))

        x_test = np.vstack(x_test).T
        pred = self.classifier.predict_proba(x_test)[:,1]
        return pred


def load_ssf(survey):
    """
    Returns appropriate survey selection function from https://arxiv.org/abs/1912.03302
    """
    config_file = '../Classifier/des_y3a2_survey_selection_function-{}-density.yaml'.format(survey)
    ssf = surveySelectionFunction(config_file)
    ssf.loadClassifier()
    ssf.load_map()
    return ssf
    

def apply_ssfs(satellite_properties,ssfs,size_cut=10.,pc_to_kpc=1000.,Mr_to_MV=-0.2,surveys=['des','ps1']):
    """
    Calculates each satellite's contribution to the observed population as (1. - disruption probability) x (observational detection probability)

    Inputs:
    	satellite_properties (dictionary of arrays): properties of mock satellites
    	ssfs (dictionary of xgboost models): dictionary containing survey selection function for each survey
    	size_cut (float): half-light radius below which objects are considered star clusters (default 10pc)
    	pc_to_kpc (float): conversion from pc to kpc
    	Mr_to_MV (float): conversion from r-band to V-band absolute magnitude

    Returns:
    	p_det (array): detection probabilities for mock satellites
    """
    p_det = np.zeros(len(satellite_properties['Mr']), dtype='f8')
    r_sat = LA.norm(satellite_properties['rotated_pos'], axis=1)
    p_surv = satellite_properties['prob']
    
    for survey in surveys:
    	flags = satellite_properties['{}_flags'.format(survey)]
    	p_det[flags] = (p_surv[flags])*ssfs[survey].predict(distance=r_sat[flags],
														    	  abs_mag=satellite_properties['Mr'][flags]+Mr_to_MV,
														    	  r_physical=satellite_properties['r12'][flags]/pc_to_kpc,
														    	  ra=satellite_properties['ra'][flags],
														    	  dec=satellite_properties['dec'][flags])
    
    #Enforce star cluster cut
    p_det[satellite_properties['r12'] < size_cut] = 0.

    return p_det