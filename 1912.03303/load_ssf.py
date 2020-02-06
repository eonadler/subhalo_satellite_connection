import yaml
import os
import healpy as hp
import xgboost as xgb

class surveySelectionFunction:
    
    def __init__(self, config_file):
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
        classifier_data = ''.join(reader.readlines())
        reader.close()
        self.classifier = xgb.XGBClassifier({'nthread': 4})
        self.classifier.load_model(self.config[self.algorithm]['classifier'])
        
    def predict(self, **kwargs):
        assert self.classifier is not None, 'ERROR'
        ra  = kwargs.pop('ra',None)
        dec = kwargs.pop('dec',None)
        nside = hpraw.get_nside(self.map)
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