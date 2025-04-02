import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != 'cpu':
  dtype = torch.cuda.FloatTensor
else:
  dtype = torch.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import warnings
import h5py
import time,sys,os,glob
from datetime import datetime

from ..train import NNmodels_new as NNmodels

def defmod(D_in,H1,H2,H3,D_out,NNtype='MLP_v0'):
    if NNtype == 'MLP_v0':
        return NNmodels.MLP_v0(D_in,H1,H2,H3,D_out)
    elif NNtype == 'MLP_v1':
        return NNmodels.MLP_v1(D_in,H1,H2,H3,D_out)

def readNN(nnpath,nntype='MLP_v0'):
    # read in the file for the previous run 
    nnh5 = h5py.File(nnpath,'r')

    if nntype == 'MLP_v0':
        D_in  = nnh5['model/mlp.lin1.weight'].shape[1]
        H1    = nnh5['model/mlp.lin1.bias'].shape[0]
        H2    = nnh5['model/mlp.lin2.bias'].shape[0]
        H3    = nnh5['model/mlp.lin3.bias'].shape[0]
        D_out = nnh5['model/mlp.lin6.bias'].shape[0]

    elif nntype == 'MLP_v1':
        D_in  = nnh5['model/mlp.lin1.weight'].shape[1]
        H1    = nnh5['model/mlp.lin1.bias'].shape[0]
        H2    = nnh5['model/mlp.lin2.bias'].shape[0]
        H3    = nnh5['model/mlp.lin3.bias'].shape[0]        
        D_out = nnh5['model/mlp.linout.bias'].shape[0]
    elif nntype == 'CNN':
        pass
    
    model = defmod(D_in,H1,H2,H3,D_out,NNtype=nntype)

    model.D_in = D_in
    model.H1 = H1
    model.H2 = H2
    model.H3 = H3
    model.D_out = D_out

    newmoddict = {}
    for kk in nnh5['model'].keys():
        nparr = nnh5['model'][kk][()]
        torarr = torch.from_numpy(nparr).type(dtype)
        newmoddict[kk] = torarr    
    model.load_state_dict(newmoddict)
    model.eval()
    nnh5.close()
    return model

class ANN(object):
    """docstring for ANN"""
    def __init__(self, nnpath=None,**kwargs):
        super(ANN, self).__init__()

        self.verbose = kwargs.get('verbose',False)

        if nnpath != None:
          self.nnpath = nnpath
        else:
            raise IOError('... Must provide a path to the ANN model')

        self.norm = kwargs.get('norm',False)

        if self.verbose:
          print('... Reading in {0}'.format(self.nnpath))

        self.nntype = kwargs.get('nntype','MLP')

        self.model = readNN(self.nnpath,nntype=self.nntype)

        # read in normalization info
        with h5py.File(self.nnpath,'r') as th5:

            self.label_i = np.array([x.decode('utf-8') for x in th5['label_i'][()]])
            self.label_o = np.array([x.decode('utf-8') for x in th5['label_o'][()]])

            if self.norm:
                self.norm_i = [th5[f'norm_i/{kk}'][()] for kk in self.label_i]
                self.norm_o = [th5[f'norm_o/{kk}'][()] for kk in self.label_o]
        

    def eval(self,x):
        
        # read in x array and check if it is one set of pars or an array of pars
        if isinstance(x,list):
            x = np.asarray(x)

        # make a copy so that the input x array isn't changed in place
        x_i = np.copy(x)

        if len(x.shape) == 1:
            inputD = 1
            if self.norm:
                for ii,n_i in enumerate(self.norm_i):
                    # x_i[ii] = 1.0 + (x_i[ii]-n_i[0])/(n_i[2]-n_i[1]) 
                    mid = n_i[0]
                    std = n_i[1]
                    x_i[ii] = (x_i[ii]-mid)/std
        else:
            inputD = x.shape[0]
            if self.norm:
                for ii,n_i in enumerate(self.norm_i):
                    # x_i[:,ii] = 1.0 + (x_i[:,ii]-n_i[0])/(n_i[2]-n_i[1]) 
                    mid = n_i[0]
                    std = n_i[1]
                    x_i[:,ii] = (x_i[:,ii]-mid)/std

        inputX = Variable(torch.from_numpy(x_i).type(dtype)).reshape(inputD,self.model.D_in)
        outputY = self.model(inputX)
        y = outputY.data.numpy().squeeze()

        if self.norm:
            if len(x.shape) == 1:
                for ii,n_i in enumerate(self.norm_o):
                    # y[ii] = ((y[ii]-1.0)*(n_i[2]-n_i[1])) + n_i[0]
                    mid = n_i[0]
                    std = n_i[1]
                    y[ii] = (y[ii]*std) + mid
            else:
                for ii,n_i in enumerate(self.norm_o):
                    # y[:,ii] = ((y[:,ii]-1.0)*(n_i[2]-n_i[1])) + n_i[0]
                    mid = n_i[0]
                    std = n_i[1]
                    y[:,ii] = (y[:,ii]*std) + mid
        return y


class modpred(object):
    """docstring for modpred"""
    def __init__(self, nnpath=None, nntype='MLP_v0', norm=False):
        super(modpred, self).__init__()
        
        if nnpath != None:
            self.nnpath = nnpath
        else:
            raise IOError('... Must provide a path to the ANN model')

        self.norm = norm

        self.anns = ANN(nnpath=self.nnpath,nntype=nntype,norm=self.norm)

        self.modpararr = self.anns.label_o
        
      
    def pred(self,inpars):
        return self.anns.eval(inpars)
    
    def getPhot(self,pars):
                
        # make copy of input array so that the code doesn't change inplace
        pars = np.copy(pars)
        
        modpred = self.pred(pars)
    
        out = {}
        
        # stick in input labels
        for ii,kk in enumerate(self.anns.label_i):
            if len(pars.shape) == 1:
                out[kk] = pars[ii] 
            else:
                out[kk] = pars[:,ii]
        
        if len(pars.shape) == 1:
            out_i = {y:modpred[ii] for ii,y in enumerate(self.anns.label_o)}
            out.update(out_i)
        else:
            out_i = {y:modpred[:,ii] for ii,y in enumerate(self.anns.label_o)}
            out.update(out_i)
        
        return out
