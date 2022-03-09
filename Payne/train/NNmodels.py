#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != 'cpu':
  dtype = torch.cuda.FloatTensor
else:
  dtype = torch.FloatTensor
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

# import numpy as np
import jax.numpy as np
import warnings
import time,sys,os,glob


# simple multi-layer perceptron model
class SMLP(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out, xmin, xmax):
        super(SMLP, self).__init__()

        self.xmin = xmin
        self.xmax = xmax

        self.features = nn.Sequential(
            nn.Linear(D_in, H1),
            nn.LeakyReLU(),
            nn.Linear(H1, H2),
            nn.LeakyReLU(),
            nn.Linear(H2, H3),
            nn.LeakyReLU(),
            nn.Linear(H3, D_out),
        )
    
    def encode(self,x):
        # convert x into numpy to do math
        x_np = x.data.cpu().numpy()
        xout = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
        return Variable(torch.from_numpy(xout).type(dtype))

    def forward(self, x):
        x_i = self.encode(x)
        return self.features(x_i)

# linear feed-foward model with sigmoid activation functions
class LinNet(nn.Module):  
    def __init__(self, D_in, H1, H2, H3, D_out, xmin, xmax):
        super(LinNet, self).__init__()

        self.xmin = xmin
        self.xmax = xmax

        self.lin1 = nn.Linear(D_in, H1)
        self.lin2 = nn.Linear(H1,H1)
        self.lin3 = nn.Linear(H1,H2)
        self.lin4 = nn.Linear(H2,H2)
        self.lin5 = nn.Linear(H2,H3)
        self.lin6 = nn.Linear(H3, D_out)

    def forward(self, x):
        x_i = self.encode(x)
        out1 = torch.sigmoid(self.lin1(x_i))
        out2 = torch.sigmoid(self.lin2(out1))
        out3 = torch.sigmoid(self.lin3(out2))
        out4 = torch.sigmoid(self.lin4(out3))
        out5 = torch.sigmoid(self.lin5(out4))
        y_i = self.lin6(out5)
        return y_i     

    def encode(self,x):
        # convert x into numpy to do math
        x_np = x.data.cpu().numpy()
        xout = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
        return Variable(torch.from_numpy(xout).type(dtype))

# ResNet convolutional neural network (two convolutional layers)
class ResNet(nn.Module):
    def __init__(self, D_in, H1, H2, D_out, xmin, xmax):
        super(ResNet, self).__init__()

        self.xmin = xmin
        self.xmax = xmax

        self.D_in = D_in
        self.H1    = H1
        self.H2    = H2
        self.D_out = D_out

        self.features = nn.Sequential(
            nn.Linear(D_in, H1),
            nn.BatchNorm1d(H1),
            nn.LeakyReLU(),
            nn.Linear(H1, H2),
            nn.LeakyReLU(),
            nn.Linear(H2, self.D_out),
        )

        kernel_size = 11
        
        self.deconv1 = nn.ConvTranspose1d(self.D_out, 64, kernel_size, stride=3, padding=5)
        self.deconv2 = nn.ConvTranspose1d(64, 64, kernel_size, stride=3, padding=5)
        self.deconv3 = nn.ConvTranspose1d(64, 64, kernel_size, stride=3, padding=5)
        self.deconv4 = nn.ConvTranspose1d(64, 64, kernel_size, stride=3, padding=5)
        self.deconv5 = nn.ConvTranspose1d(64, 64, kernel_size, stride=3, padding=5)
        self.deconv6 = nn.ConvTranspose1d(64, 32, kernel_size, stride=3, padding=5)
        self.deconv7 = nn.ConvTranspose1d(32, 1,  kernel_size, stride=3, padding=5)

        self.deconv2b = nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv3b = nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv4b = nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv5b = nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv6b = nn.ConvTranspose1d(64, 32, 1, stride=3)

        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        self.relu4 = nn.LeakyReLU()
        self.relu5 = nn.LeakyReLU()
        self.relu6 = nn.LeakyReLU()

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)


    def forward(self, x):
        x_i = self.encode(x)
        x_i = self.features(x_i)#[:,None,:]
        # print(x_i.shape)
        # x_i = x_i.view(x_i.shape[0], 1, self.D_out)

        x1 = self.deconv1(x_i)

        x2 = self.deconv2(x1)
        x2 += self.deconv2b(x1)
        x2 = self.relu2(x2)
        x2 = self.dropout1(x2)

        x3 = self.deconv3(x2)
        x3 += self.deconv3b(x2)
        x3 = self.relu2(x3)
        x3 = self.dropout1(x3)

        x4 = self.deconv4(x3)
        x4 += self.deconv4b(x3)
        x4 = self.relu2(x4)
        x4 = self.dropout2(x4)

        x5 = self.deconv5(x4)
        x5 += self.deconv5b(x4)
        x5 = self.relu2(x5)
        x5 = self.dropout2(x5)

        x6 = self.deconv6(x5)
        x6 += self.deconv6b(x5)
        x6 = self.relu2(x6)
        x6 = self.dropout2(x6)

        x7 = self.deconv7(x6)[:,0,:self.D_out]

        return x7


    def encode(self,x):
        # convert x into numpy to do math
        x_np = x.data.cpu().numpy()
        xout = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
        return Variable(torch.from_numpy(xout).type(dtype))

# class Net(object):
#      def __init__(self, NNpath):
#           self.readNN(nnpath=NNpath)

#      def readNN(self,nnpath=''):

#           th5 = h5py.File(nnpath,'r')
#           self.w_array_0 = np.array(th5['w_array_0'])
#           self.w_array_1 = np.array(th5['w_array_1'])
#           self.w_array_2 = np.array(th5['w_array_2'])
#           self.b_array_0 = np.array(th5['b_array_0'])
#           self.b_array_1 = np.array(th5['b_array_1'])
#           self.b_array_2 = np.array(th5['b_array_2'])
#           self.xmin      = np.array(th5['x_min'])
#           self.xmax      = np.array(th5['x_max'])

#           self.wavelength = np.array(th5['wavelength'])

#           th5.close()

#           self.resolution = 100000


#      def leaky_relu(self,z):
#          '''
#          This is the activation function used by default in all our neural networks.
#          '''
#          return z*(z > 0) + 0.01*z*(z < 0)
     
#      def encode(self,x):
#           x_np = np.array(x)
#           x_scaled = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
#           return x_scaled

#      def eval(self,x):
#           x_i = self.encode(x)
#           inside = np.einsum('ij,j->i', self.w_array_0, x_i) + self.b_array_0
#           outside = np.einsum('ij,j->i', self.w_array_1, self.leaky_relu(inside)) + self.b_array_1
#           modspec = np.einsum('ij,j->i', self.w_array_2, self.leaky_relu(outside)) + self.b_array_2

#           return modspec