from ..predict.photANN_new import modpred
import h5py
import numpy as np

class TestPhot(object):
    def __init__(self, nnpath=None, nntype='MLP', norm=False):
        super(TestPhot, self).__init__()
        
        if nnpath != None:
            self.nnpath = nnpath
        else:
            raise IOError('... Must provide a path to the ANN model')

        self.norm = norm

        self.model = modpred(
            nnpath=self.nnpath, 
            nntype=nntype,
            norm=self.norm,)

        # pull out test labels
        with h5py.File(self.nnpath,'r') as h5:
            self.testlabels_in  = h5['testlabels_in'][()]
            self.testlabels_out = h5['testlabels_out'][()]
            
        
    def test(self):
        print('Testing photANN')
        
        # generate model predictions based on test input labels
        mod_pred = self.model.getPhot(self.testlabels_in)
        
        