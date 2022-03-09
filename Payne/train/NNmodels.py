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
import warnings
with warnings.catch_warnings():
     warnings.simplefilter('ignore')
     import h5py
import time,sys,os,glob
from datetime import datetime

from scipy import constants
speedoflight = constants.c / 1000.0
import numpy as np

import Payne

# Read in various types of saved NN
def defmod(D_in,H1,H2,H3,D_out,xmin,xmax,NNtype='SMLP'):
    if NNtype == 'ResNet':
        return ResNet(D_in,H1,H2,H3,D_out,xmin,xmax)
    elif NNtype == 'LinNet':
        return LinNet(D_in,H1,H2,H3,D_out,xmin,xmax)
    else:
        return SMLP(D_in,H1,H2,H3,D_out,xmin,xmax)

def readNN(nnpath,NNtype='SMLP'):
    # read in the file for the previous run 
    nnh5 = h5py.File(nnpath,'r')

    xmin  = nnh5['xmin'][()]
    xmax  = nnh5['xmax'][()]

    if (NNtype == 'SMLP'):
        D_in  = nnh5['model/features.0.weight'].shape[1]
        H1    = nnh5['model/features.0.weight'].shape[0]
        H2    = nnh5['model/features.2.weight'].shape[0]
        H3    = nnh5['model/features.4.weight'].shape[0]
        D_out = nnh5['model/features.6.weight'].shape[0]

    if (NNtype == 'LinNet'):
        D_in  = nnh5['model/lin1.weight'].shape[1]
        H1    = nnh5['model/lin1.weight'].shape[0]
        H2    = nnh5['model/lin4.weight'].shape[0]
        H3    = nnh5['model/lin5.weight'].shape[0]
        D_out = nnh5['model/lin6.weight'].shape[0]
    
    if NNtype == 'ResNet':
        D_in      = nnh5['model/ConvTranspose1d.weight'].shape[1]
        H1        = nnh5['model/lin1.weight'].shape[0]
        H2        = nnh5['model/lin2.weight'].shape[0]
        H3        = nnh5['model/lin3.weight'].shape[0]
        outputdim = len([x_i for x_i in list(nnh5['model'].keys()) if 'weight' in x_i])
        D_out     = nnh5['model/lin{0}.weight'.format(outputdim)].shape[0]
    
    model = defmod(D_in,H1,H2,H3,D_out,xmin,xmax,NNtype=NNtype)

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

     def leaky_relu(self,z):
          '''
          This is the activation function used by default in all our neural networks.
          '''
          return z*(z > 0) + 0.01*z*(z < 0)

     def npencode(self,x):
          # convert x into numpy to do math
          x_np = np.array(x)
          xout = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
          return xout

     def npeval(self,x):
          x_i = self.npencode(x)
          L1      = np.einsum('ij,j->i', self.features[0].weight.detach().numpy(), x_i)                 + self.features[0].bias.detach().numpy()
          L2      = np.einsum('ij,j->i', self.features[2].weight.detach().numpy(), self.leaky_relu(L1)) + self.features[2].bias.detach().numpy()
          L3      = np.einsum('ij,j->i', self.features[4].weight.detach().numpy(), self.leaky_relu(L2)) + self.features[4].bias.detach().numpy()
          modspec = np.einsum('ij,j->i', self.features[6].weight.detach().numpy(), self.leaky_relu(L3)) + self.features[6].bias.detach().numpy()
          return modspec

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