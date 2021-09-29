#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the code that predicts photometry given set of Teff/log(g)/[Fe/H]/[a/Fe]
"""

import jax.numpy as np
import math
import torch
from torch import nn
dtype = torch.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F
import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  import h5py
from itertools import product
import copy

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
    return Variable(torch.from_numpy(x.copy()).type(dtype))

class ANN(object):
  """docstring for nnBC"""
  def __init__(self, ff, nnpath=None,**kwargs):
    super(ANN, self).__init__()

    self.verbose = kwargs.get('verbose',True)

    if nnpath != None:
      self.nnpath = nnpath
    else:
      # define aliases for the MIST isochrones and C3K/CKC files
      self.nnpath  = Payne.__abspath__+'data/photANN/'

    self.nnh5 = self.nnpath+'nnMIST_{0}.h5'.format(ff)

    if self.verbose:
      print('... Phot ANN: Reading in {0}'.format(self.nnh5))
    th5 = h5py.File(self.nnh5,'r')
    
    self.D_in = th5['model/lin1.weight'].shape[1]
    self.H = th5['model/lin1.weight'].shape[0]
    self.D_out = th5['model/lin3.weight'].shape[0]
    self.model = Net(self.D_in,self.H,self.D_out)
    # self.model.xmin = np.amin(np.array(th5['test/X']),axis=0)
    # self.model.xmax = np.amax(np.array(th5['test/X']),axis=0)
    self.model.xmin = np.array(list(th5['xmin']))#np.amin(np.array(th5['test/X']),axis=0)
    self.model.xmax = np.array(list(th5['xmax']))#np.amax(np.array(th5['test/X']),axis=0)

    newmoddict = {}
    for kk in th5['model'].keys():
      nparr = copy.deepcopy(np.array(th5['model'][kk]))
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

    inputVar = Variable(torch.from_numpy(x.copy()).type(dtype)).resize(inputD,self.D_in)
    outpars = self.model(inputVar)
    return outpars.data.numpy().squeeze()


class fastANN(object):

    def __init__(self, nnlist, bandlist):
        self.filternames = bandlist
        self.w1 = np.array([nn.model.lin1.weight.data.numpy() for nn in nnlist])
        self.b1 = np.expand_dims(np.array([nn.model.lin1.bias.data.numpy() for nn in nnlist]), -1)
        self.w2 = np.array([nn.model.lin2.weight.data.numpy() for nn in nnlist])
        self.b2 = np.expand_dims(np.array([nn.model.lin2.bias.data.numpy() for nn in nnlist]), -1)
        self.w3 = np.array([nn.model.lin3.weight.data.numpy() for nn in nnlist])
        self.b3 = np.expand_dims(np.array([nn.model.lin3.bias.data.numpy() for nn in nnlist]), -1)

        self.set_minmax(nnlist[0])

    def set_minmax(self, nn):
      try:
        self.xmin = nn.model.xmin
        self.xmax = nn.model.xmax
      except (NameError,AttributeError):
        self.xmin = np.amin(nn.model.x.data.numpy(),axis=0)
        self.xmax = np.amax(nn.model.x.data.numpy(),axis=0)

      self.range = (self.xmax - self.xmin)

    def encode(self, x):
        xp = (np.atleast_2d(x) - self.xmin) / self.range
        return xp.T

    def sigmoid(self, a):
        return 1. / (1 + np.exp(-a))

    def eval(self, x):
        """With some annying shape changes
        """
        a1 = self.sigmoid(np.matmul(self.w1,  self.encode(x)) + self.b1)
        a2 = self.sigmoid(np.matmul(self.w2, a1) + self.b2)
        y = np.matmul(self.w3, a2) + self.b3
        return np.squeeze(y)

