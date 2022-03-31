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

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import Payne

from ..utils.readc3k import readc3k
from ..utils import optim
from . import NNmodels
from ..train.NNmodels import readNN

fwhm_to_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))

def slicebatch(inlist,N):
     '''
     Function to slice a list into batches of N elements. Last sublist might have < N elements.
     '''
     return [inlist[ii:ii+N] for ii in range(0,len(inlist),N)]

def defmod(D_in,H1,H2,H3,D_out,xmin,xmax,NNtype='SMLP'):
     if NNtype == 'ResNet':
          return NNmodels.ResNet(D_in,H1,H2,D_out,xmin,xmax)
     elif NNtype == 'LinNet':
          return NNmodels.LinNet(D_in,H1,H2,H3,D_out,xmin,xmax)
     else:
          return NNmodels.SMLP(D_in,H1,H2,H3,D_out,xmin,xmax)

class TrainMod(object):
     """docstring for TrainMod"""
     def __init__(self, *arg, **kwargs):
          super(TrainMod, self).__init__()

          # number of models to train on
          if 'numtrain' in kwargs:
               self.numtrain = kwargs['numtrain']
          else:
               self.numtrain = 20000

          if 'numtest' in kwargs:
               self.numtest = kwargs['numtest']
          else:
               self.numtest = int(0.1*self.numtrain)

          if 'numsteps' in kwargs:
               self.numsteps = kwargs['numsteps']
          else:
               self.numsteps = int(1e+4)

          if 'numepochs' in kwargs:
               self.numepochs = kwargs['numepochs']
          else:
               self.numepochs = 1

          if 'batchsize' in kwargs:
               self.batchsize = kwargs['batchsize']
          else:
               self.batchsize = self.numtrain

          # number of nuerons in each layer
          if 'H1' in kwargs:
               self.H1 = kwargs['H1']
          else:
               self.H1 = 256

          if 'H2' in kwargs:
               self.H2 = kwargs['H2']
          else:
               self.H2 = 256

          if 'H3' in kwargs:
               self.H3 = kwargs['H3']
          else:
               self.H3 = 256

          # set user defined labels
          if 'labels_in' in kwargs:
               self.label_i = kwargs['labels_in']
          else:
               self.label_i = ['teff','logg','feh','afe']

          resolution_fwhm = kwargs.get('resolution',32000.0)
          # convert resolution to sigma
          self.resolution = resolution_fwhm * fwhm_to_sigma

          self.waverange  = kwargs.get('waverange',[5150.0,5300.0])

          # check for user defined ranges for atm models
          self.teffrange  = kwargs.get('teff',None)
          self.loggrange  = kwargs.get('logg',None)
          self.fehrange   = kwargs.get('FeH',None)
          self.aferange   = kwargs.get('aFe',None)
          self.vtrange    = kwargs.get('vturb',None)

          if self.vtrange is 'fixed':
               vtfixed = True
          else:
               vtfixed = False

          self.restartfile = kwargs.get('restartfile',False)
          if self.restartfile is not False:
               print('... Restarting File: {0}'.format(self.restartfile))

          # output hdf5 file name
          self.outfilename = kwargs.get('output','TESTOUT.h5')

          # path to C3K lib
          self.c3kpath  = kwargs.get('c3kpath',None)

          # path to MIST lib
          self.mistpath = kwargs.get('mistpath',None)

          # type of NN to train
          self.NNtype = kwargs.get('NNtype','SMLP')

          # turn on log plotting, could be memory intensive
          self.logplot = kwargs.get('logplot',False)

          # starting learning rate
          self.lr = kwargs.get('lr',1E-4)

          startreadintestmod = datetime.now()
          # initialzie class to pull models
          print('... Pulling a first set of models for test set')
          print('... Reading {0} test models from '.format(self.numtest))
          print('    c3k: {0}'.format(self.c3kpath))
          print('    mist: {0}'.format(self.mistpath))
          self.c3kmods = readc3k(MISTpath=self.mistpath,C3Kpath=self.c3kpath,vtfixed=vtfixed)
          sys.stdout.flush()

          spectra_test,labels_test,wavelength_test = self.c3kmods.pullspectra(
               self.numtest,
               resolution=self.resolution, 
               waverange=self.waverange,
               MISTweighting=True,
               Teff=self.teffrange,
               logg=self.loggrange,
               FeH=self.fehrange,
               aFe=self.aferange,
               vtrub=self.vtrange)

          self.testlabels = labels_test[:,:len(self.label_i)].tolist()

          # # pull a quick set of test models to determine general properties
          # mod_test = self.mistmods.pullmod(
          #     self.numtest,
          #     eep=self.eeprange,
          #     mass=self.massrange,feh=self.FeHrange,afe=self.aFerange)

          # self.mod_test = Table()
          # for x in mod_test.keys():
          #     if x != 'label_o':
          #         self.mod_test[x] = mod_test[x]
          # self.testlabels = mod_test['label_o']

          print('... Finished reading in test set of models ({0})'.format(datetime.now() - startreadintestmod))

          # determine normalization values
          self.xmin = np.array([self.c3kmods.minmax[x][0] 
               for x in self.label_i])
          self.xmax = np.array([self.c3kmods.minmax[x][1] 
               for x in self.label_i])
          self.ymin = np.array([self.c3kmods.Fminmax[0]])
          self.ymax = np.array([self.c3kmods.Fminmax[1]])

          # D_in is input dimension
          # D_out is output dimension
          self.D_in  = len(self.label_i)
          self.D_out = len(wavelength_test)

          # initialize the output file
          with h5py.File('{0}'.format(self.outfilename),'w') as outfile_i:
               try:
                    outfile_i.create_dataset('testpred',
                         data=np.array(spectra_test),compression='gzip')
                    outfile_i.create_dataset('testlabels',
                         data=self.testlabels,compression='gzip')
                    outfile_i.create_dataset('label_i',
                         data=np.array([x.encode("ascii", "ignore") for x in self.label_i]))
                    outfile_i.create_dataset('wavelengths',
                         data=np.array(wavelength_test))
                    outfile_i.create_dataset('resolution',data=np.array(self.resolution))
                    outfile_i.create_dataset('xmin',data=np.array(self.xmin))
                    outfile_i.create_dataset('xmax',data=np.array(self.xmax))
                    outfile_i.create_dataset('ymin',data=np.array(self.ymin))
                    outfile_i.create_dataset('ymax',data=np.array(self.ymax))
               except:
                    print('!!! PROBLEM WITH WRITING TO HDF5 !!!')
                    raise

          print('... Din: {}, Dout: {}'.format(self.D_in,self.D_out))
          print('... Input Labels: {}'.format(self.label_i))
          print('... Output WL Range: {0} - {1}'.format(
               wavelength_test.min(),
               wavelength_test.max(),))
          print('... label min values:')
          print('    {}'.format(self.xmin))
          print('... label max values:')
          print('    {}'.format(self.xmax))
          print('... Resolution FWHM: {0}'.format(resolution_fwhm))
          print('... Resolution Sigma: {0}'.format(self.resolution))

          print('... Finished Init')
          sys.stdout.flush()

     def __call__(self):
          '''
          call instance so that train_pixel can be called with multiprocessing
          and still have all of the class instance variables

          '''
          try:
               return self.train_mod()
          except Exception as e:
               traceback.print_exc()
               print()
               raise e

     def run(self):
          '''
          function to actually run the training on models

          '''
          # start total timer
          tottimestart = datetime.now()

          print('Starting Training at {0}'.format(tottimestart))
          sys.stdout.flush()

          net = self()
          if type(net[0]) == type(None):
               return net

     def train_mod(self):
          '''
          function to train the network
          '''
          # start a timer
          starttime = datetime.now()

          if str(device) != 'cpu':
               # determine if this is running within mp
               if len(multiprocessing.current_process()._identity) > 0:
                    torch.cuda.set_device(multiprocessing.current_process()._identity[0]-1)

               print('Running on GPU: {0}/{1}'.format(
                    torch.cuda.current_device()+1,
                    torch.cuda.device_count(),
                    ))


          # determine if user wants to start from old file, or
          # create a new ANN model
          if self.restartfile is not False:
               # create a model
               if os.path.isfile(self.restartfile):
                    print('Restarting from File: {0} with NNtype: {1}'.format(self.restartfile,self.NNtype))
                    sys.stdout.flush()
                    model = readNN(self.restartfile,NNtype=self.NNtype)
               else:
                    print('Could Not Find Restart File, Creating a New NN model')
                    sys.stdout.flush()
                    model = defmod(self.D_in,self.H1,self.H2,self.H3,self.D_out,
                         self.xmin,self.xmax,NNtype=self.NNtype)        
          else:
               # initialize the model
               print('Running New NN with NNtype: {0}'.format(self.NNtype))
               sys.stdout.flush()
               model = defmod(self.D_in,self.H1,self.H2,self.H3,self.D_out,
                    self.xmin,self.xmax,NNtype=self.NNtype)

          # set up model to start training
          model.to(device)
          model.train()

          # initialize the loss function
          loss_fn = torch.nn.MSELoss(reduction='sum')
          # loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
          # loss_fn = torch.nn.KLDivLoss(size_average=False)
          # loss_fn = torch.nn.L1Loss(reduction = 'sum')

          # initialize the optimizer
          learning_rate = self.lr
          # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
          # optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
          optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
          # we adopt rectified Adam for the optimization
          # optimizer = radam.RAdam(
          #     [p for p in model.parameters() if p.requires_grad==True], lr=learning_rate)

          # initialize the scheduler to adjust the learning rate
          scheduler = StepLR(optimizer,100,gamma=0.90)
          # scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.1)
          # scheduler = ExponentialLR(optimizer,100,gamma=0.90)

          # number of batches
          nbatches = self.numtrain // self.batchsize

          print('... Number of epochs: {}'.format(self.numepochs))
          print('... Number of training steps: {}'.format(self.numsteps))
          print('... Number of models in each batch: {}'.format(self.batchsize))
          print('... Number of batches: {}'.format(nbatches))

          # cycle through epochs
          for epoch_i in range(int(self.numepochs)):
               epochtime = datetime.now()
               print('... Pulling {0} Training/Validation Models for Epoch: {1}'.format(self.numtrain,epoch_i+1))
               sys.stdout.flush()

               # initiate counter
               current_loss = np.inf
               iter_arr = []
               training_loss =[]
               validation_loss = []
               medres_loss = []
               maxres_loss = []

               startreadintrainmod = datetime.now()

               spectra_train,labels_train,wavelength_train = self.c3kmods.pullspectra(
                    self.numtrain,
                    resolution=self.resolution, 
                    waverange=self.waverange,
                    MISTweighting=True,
                    Teff=self.teffrange,
                    logg=self.loggrange,
                    FeH=self.fehrange,
                    aFe=self.aferange,
                    vtrub=self.vtrange,
                    excludelabels=np.array(self.testlabels),
                    )

               # create tensor for input training labels
               X_train_labels = labels_train[:,:len(self.label_i)]
               X_train_Tensor = Variable(torch.from_numpy(X_train_labels).type(dtype))
               X_train_Tensor = X_train_Tensor.to(device)

               # create tensor of output training labels
               Y_train = np.array(spectra_train)
               Y_train_Tensor = Variable(torch.from_numpy(Y_train).type(dtype), requires_grad=False)
               Y_train_Tensor = Y_train_Tensor.to(device)

               spectra_valid,labels_valid,wavelength_valid = self.c3kmods.pullspectra(
                    self.numtrain,
                    resolution=self.resolution, 
                    waverange=self.waverange,
                    MISTweighting=True,
                    Teff=self.teffrange,
                    logg=self.loggrange,
                    FeH=self.fehrange,
                    aFe=self.aferange,
                    vtrub=self.vtrange,
                    excludelabels=np.array(list(self.testlabels)+list(X_train_labels)),)

               # create tensor for input validation labels
               X_valid_labels = labels_valid[:,:len(self.label_i)]
               X_valid_Tensor = Variable(torch.from_numpy(X_valid_labels).type(dtype))
               X_valid_Tensor = X_valid_Tensor.to(device)

               # create tensor of output validation labels
               Y_valid = np.array(spectra_valid)
               Y_valid_Tensor = Variable(torch.from_numpy(Y_valid).type(dtype), requires_grad=False)
               Y_valid_Tensor = Y_valid_Tensor.to(device)

               print('... Finished reading in models ({})'.format(datetime.now() - startreadintrainmod))

               cc = 0
               for iter_i in range(int(self.numsteps)):

                    itertime = datetime.now()

                    perm = torch.randperm(self.numtrain)

                    if str(device) != 'cpu':
                         perm = perm.cuda()

                    for t in range(nbatches):
                         steptime = datetime.now()

                         idx = perm[t * self.batchsize : (t+1) * self.batchsize]
                         def closure():
                              # Forward pass: compute predicted y by passing x to the model.
                              Y_pred_train_Tensor = model(X_train_Tensor[idx])

                              # Compute and print loss.
                              loss = loss_fn(Y_pred_train_Tensor, Y_train_Tensor[idx])

                              # Backward pass: compute gradient of the loss with respect to model parameters
                              optimizer.zero_grad()
                              loss.backward(retain_graph=False)
                              # optimizer.step()
                             
                              if np.isnan(loss.item()):
                                   print('PRED TRAIN TENSOR',Y_pred_train_Tensor)
                                   print('TRAIN TENSOR',Y_train_Tensor)
                                   return loss
                              return loss

                         # Calling the step function on an Optimizer makes an update to its parameters
                         loss = optimizer.step(closure)

                         # Y_pred_train_Tensor = model(X_train_Tensor[idx])
                         # loss = loss_fn(Y_pred_train_Tensor, Y_train_Tensor[idx])
                         # optimizer.zero_grad()
                         # loss.backward(retain_graph=False)
                         # optimizer.step()

                    # evaluate the validation set
                    if iter_i % 100 == 0:
                         perm_valid = torch.randperm(self.numtrain)
                         if str(device) != 'cpu':
                              perm_valid = perm_valid.cuda()

                         loss_valid = 0
                         medres = 0
                         maxres = -np.inf
                         for j in range(nbatches):
                              idx = perm[t * self.batchsize : (t+1) * self.batchsize]

                              Y_pred_valid_Tensor = model(X_valid_Tensor[idx]) 
                              loss_valid += loss_fn(Y_pred_valid_Tensor, Y_valid_Tensor[idx])
                              if self.logplot:
                                   residual = torch.abs(Y_pred_valid_Tensor-Y_valid_Tensor[idx])
                                   medres_i,maxres_i = float(residual.median()),float(residual.max())
                                   if medres_i > medres:
                                        medres = medres_i
                                   if maxres_i > maxres:
                                        maxres = maxres_i

                         loss_valid /= nbatches

                         loss_data = loss.detach().data.item()
                         loss_valid_data = loss_valid.detach().data.item()

                         if self.logplot:
                              iter_arr.append(iter_i)
                              training_loss.append(loss_data)
                              validation_loss.append(loss_valid_data)
                              medres_loss.append(medres)
                              maxres_loss.append(maxres)

                              fig,ax = plt.subplots(nrows=3,ncols=1)
                              ax[0].plot(iter_arr,np.log10(training_loss) - np.log10(len(wavelength_valid)),ls='-',lw=1.0,alpha=0.75,c='C0',label='Training')
                              ax[0].plot(iter_arr,np.log10(validation_loss) - np.log10(len(wavelength_valid)),ls='-',lw=1.0,alpha=0.75,c='C3',label='Validation')
                              ax[0].legend()
                              # ax[0].set_xlabel('Iteration')
                              ax[0].set_ylabel('log(Loss per pixel)')

                              ax[1].plot(iter_arr,np.log10(maxres_loss),ls='-',lw=1.0,alpha=0.75,c='C4',label='max')
                              ax[1].set_ylabel('log(|Max Residual|)')

                              ax[2].plot(iter_arr,np.log10(medres_loss),ls='-',lw=1.0,alpha=0.75,c='C2',label='median')
                              ax[2].set_xlabel('Iteration')
                              ax[2].set_ylabel('log(|Med Residual|)')


                              # for spec_i in Y_pred_valid_Tensor.to('cpu').numpy():
                              #      ax[1].plot(
                              #           wavelength_valid,
                              #           spec_i,
                              #           ls='-',lw=0.5)
                              #      ax[1].set_xlabel('Wavelength')
                              #      ax[1].set_ylabel('Flux')
                              fig.savefig('loss_epoch{0}.png'.format(epoch_i+1),dpi=150)
                              plt.close(fig)

                         if iter_i % 500 == 0.0:
                              print(
                                   '--> Ep: {0:d} -- Iter {1:d}/{2:d} -- Time/step: {3} -- Train Loss: {4:.6f} -- Valid Loss: {5:.6f}'.format(
                                   int(epoch_i+1),int(iter_i+1),int(self.numsteps), (datetime.now()-itertime), loss_data, loss_valid_data)
                                   )
                         sys.stdout.flush()                      


                    # # check if network has converged
                    # if np.abs(np.nan_to_num(loss_valid_data)-np.nan_to_num(current_loss))/np.abs(loss_valid_data) < 0.01:
                    #     # start counter
                    #     cc  = cc + 1

                    # if cc == 100:
                    #     print(loss_valid_data,current_loss)
                    #     current_loss = loss_valid_data
                    #     break

               # adjust the optimizer lr
               scheduler.step()
            
               # After Each Epoch, write network to output HDF5 file to save progress
               with h5py.File('{0}'.format(self.outfilename),'r+') as outfile_i:
                    for kk in model.state_dict().keys():
                         try:
                              del outfile_i['model/{0}'.format(kk)]
                         except KeyError:
                              pass

                         outfile_i.create_dataset(
                              'model/{0}'.format(kk),
                              data=model.state_dict()[kk].cpu().numpy(),
                              compression='gzip')
               print('Finished Epoch {0} @ {1} ({2})'.format(epoch_i+1, datetime.now(),datetime.now() - epochtime))

          print('Finished training model, took: {0}'.format(
               datetime.now()-starttime))
          sys.stdout.flush()

          return [model, optimizer, datetime.now()-starttime]


