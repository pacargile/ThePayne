#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the code that predicts photometry given set of Teff/log(g)/[Fe/H]/[a/Fe]
"""

import torch
from torch import nn
dtype = torch.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import h5py

import Payne

class Net(nn.Module):  
  def __init__(self, D_in, H, D_out):
    super(Net, self).__init__()
    self.lin1 = nn.Linear(D_in, H)
    self.lin2 = nn.Linear(H,H)
    self.lin3 = nn.Linear(H, D_out)

  def forward(self, x):
    x_i = self.encode(x)
    out1 = F.sigmoid(self.lin1(x_i))
    out2 = F.sigmoid(self.lin2(out1))
    y_i = self.lin3(out2)
    return y_i     

  def encode(self,x):
    try:
      self.xmin
      self.xmax
    except (NameError,AttributeError):
      self.xmin = np.amin(x.data.numpy(),axis=0)
      self.xmax = np.amax(x.data.numpy(),axis=0)

    x = (x.data.numpy()-self.xmin)/(self.xmax-self.xmin)
    return Variable(torch.from_numpy(x).type(dtype))

class ANN(object):
  """docstring for nnBC"""
  def __init__(self, ff, nnpath=None,**kwargs):
    super(ANN, self).__init__()

    self.verbose = kwargs.get('verbose',True)

    if nnpath != None:
      self.nnpath = nnpath
    else:
      # define aliases for the MIST isochrones and C3K/CKC files
      self.nnpath  = Payne.__abspath__+'data/nnMIST/'

    self.nnpath = nnpath
    self.nnh5 = self.nnpath+'nnMIST_{0}.h5'.format(ff)

    if self.verbose:
      print('... Phot ANN: Reading in {0}'.format(self.nnh5))
    th5 = h5py.File(self.nnh5,'r')
    
    D_in = th5['model/lin1.weight'].shape[1]
    H = th5['model/lin1.weight'].shape[0]
    D_out = th5['model/lin3.weight'].shape[0]
    self.model = Net(D_in,H,D_out)
    self.model.xmin = np.amin(np.array(th5['test/X']),axis=0)
    self.model.xmax = np.amax(np.array(th5['test/X']),axis=0)

    newmoddict = {}
    for kk in th5['model'].keys():
      nparr = np.array(th5['model'][kk])
      torarr = torch.from_numpy(nparr).type(dtype)
      newmoddict[kk] = torarr    
    self.model.load_state_dict(newmoddict)

  def eval(self,x):
    if type(x) == type([]):
      x = np.array(x)
    if len(x.shape) == 1:
      inputD = 1
    else:
      inputD = x.shape[0]

    inputVar = Variable(torch.from_numpy(x).type(dtype)).resize(inputD,4)
    outpars = self.model(inputVar)
    return outpars.data.numpy().squeeze()
