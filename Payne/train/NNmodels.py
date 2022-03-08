#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau

import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Pool

# from multiprocessing import Pool

import traceback
import numpy as np
import warnings
with warnings.catch_warnings():
     warnings.simplefilter('ignore')
     import h5py
import time,sys,os,glob
from datetime import datetime
try:
     # Python 2.x
     from itertools import imap
except ImportError:
     # Python 3.x
     imap=map
from scipy import constants
speedoflight = constants.c / 1000.0

import Payne

class Net(object):
     def __init__(self, NNpath):
          self.readNN(nnpath=NNpath)

     def readNN(self,nnpath=''):

          th5 = h5py.File(nnpath,'r')
          self.w_array_0 = np.array(th5['w_array_0'])
          self.w_array_1 = np.array(th5['w_array_1'])
          self.w_array_2 = np.array(th5['w_array_2'])
          self.b_array_0 = np.array(th5['b_array_0'])
          self.b_array_1 = np.array(th5['b_array_1'])
          self.b_array_2 = np.array(th5['b_array_2'])
          self.xmin      = np.array(th5['x_min'])
          self.xmax      = np.array(th5['x_max'])

          self.wavelength = np.array(th5['wavelength'])

          th5.close()

          self.resolution = 100000


     def leaky_relu(self,z):
         '''
         This is the activation function used by default in all our neural networks.
         '''
         return z*(z > 0) + 0.01*z*(z < 0)
     
     def encode(self,x):
          x_np = np.array(x)
          x_scaled = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
          return x_scaled

     def eval(self,x):
          x_i = self.encode(x)
          inside = np.einsum('ij,j->i', self.w_array_0, x_i) + self.b_array_0
          outside = np.einsum('ij,j->i', self.w_array_1, self.leaky_relu(inside)) + self.b_array_1
          modspec = np.einsum('ij,j->i', self.w_array_2, self.leaky_relu(outside)) + self.b_array_2

          return modspec