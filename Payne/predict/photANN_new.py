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
import time,sys,os,glob,re
from datetime import datetime

from ..train import NNmodels_new as NNmodels

def defmod(D_in,H1,H2,H3,D_out,NNtype='MLP_v0'):
    if NNtype == 'MLP_v0':
        return MLP_v0(D_in,H1,H2,H3,D_out)
    elif NNtype == 'MLP_v1':
        return MLP_v1(D_in,H1,H2,H3,D_out)
    elif NNtype == 'MLP_v2':
        return MLP_v2(D_in,H1,H2,H3,D_out)
    else:
        raise ValueError(f"Unknown NNtype: {NNtype}")

def _get_shape(h5grp, key):
    if key in h5grp:
        return h5grp[key].shape
    return None

def readNN(nnpath, nntype='MLP_v0'):
    """
    Load a saved MLP (v0/v1/v2) from an HDF5 checkpoint written by your trainer.
    Works with both old files (e.g., 'mlp.lin6.bias') and newer files
    (e.g., 'mlp.linout.bias'). Ignores extra groups like /split and /norms.
    """
    with h5py.File(nnpath, 'r') as nnh5:
        if 'model' not in nnh5:
            raise KeyError(f"'{nnpath}' has no '/model' group.")
        g = nnh5['model']

        # ---- Find canonical layer names present in this file ----
        # First linear layer (for D_in)
        # Prefer 'mlp.lin1.weight'; fall back to any '*.weight' if needed
        lin1_w_key = None
        for cand in ('mlp.lin1.weight',):
            if cand in g: lin1_w_key = cand; break
        if lin1_w_key is None:
            # very robust fallback: pick the first *.weight that looks 2D
            for k in g.keys():
                if k.endswith('.weight') and len(g[k].shape) == 2:
                    lin1_w_key = k; break
        if lin1_w_key is None:
            raise KeyError("Could not find a 2D weight tensor for the first linear layer.")

        D_in = g[lin1_w_key].shape[1]

        # Hidden layer sizes: try standard keys, else None
        def _size_of_bias(name):
            shp = _get_shape(g, name)
            return shp[0] if (shp is not None and len(shp) == 1) else None

        H1 = _size_of_bias('mlp.lin1.bias')
        H2 = _size_of_bias('mlp.lin2.bias')
        H3 = _size_of_bias('mlp.lin3.bias')

        # Output size: prefer 'linout.bias', fall back to highest-numbered 'lin*.bias'
        D_out = _size_of_bias('mlp.linout.bias')
        if D_out is None:
            # old files may have 'mlp.lin6.bias' (or similar)
            lin_bias = []
            for k in g.keys():
                m = re.match(r'^mlp\.lin(\d+)\.bias$', k)
                if m:
                    idx = int(m.group(1))
                    lin_bias.append((idx, k))
            if lin_bias:
                # take the largest index as the output bias
                lin_bias.sort()
                D_out = g[lin_bias[-1][1]].shape[0]

        if any(v is None for v in (H1, H2, H3, D_out)):
            raise ValueError(
                f"Could not infer full architecture from HDF5 keys. "
                f"Found: D_in={D_in}, H1={H1}, H2={H2}, H3={H3}, D_out={D_out}."
            )

        # ---- Construct the model of the requested type with inferred sizes ----
        model = defmod(D_in, H1, H2, H3, D_out, NNtype=nntype)

        # attach shapes on the module (optional, keeps your original behavior)
        model.D_in, model.H1, model.H2, model.H3, model.D_out = D_in, H1, H2, H3, D_out

        # ---- Build a state_dict from HDF5 and load strictly ----
        newmoddict = {}
        for kk in g.keys():
            # HDF5 datasets -> torch.float32 tensor on CPU
            nparr = g[kk][()]
            tens = torch.from_numpy(nparr).to(torch.float32)
            newmoddict[kk] = tens

        # This should be strict; if it fails, it means NNtype mismatches the file contents.
        model.load_state_dict(newmoddict, strict=True)
        model.eval()
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
