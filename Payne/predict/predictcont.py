#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import warnings
from datetime import datetime
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import h5py
from scipy import constants
speedoflight = constants.c / 1000.0

import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != 'cpu':
  dtype = torch.cuda.FloatTensor
else:
  dtype = torch.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F

import Payne

from ..utils.smoothing import smoothspec
from ..train.NNmodels import readNN

class ANN(object):
  """docstring for ANN"""
  def __init__(self, nnpath=None,**kwargs):
    super(ANN, self).__init__()

    self.verbose = kwargs.get('verbose',False)

    if nnpath != None:
      self.nnpath = nnpath
    else:
      self.nnpath  = Payne.__abspath__+'data/ANN/NN.h5'

    if self.verbose:
      print('... Reading in {0}'.format(self.nnpath))
    th5 = h5py.File(self.nnpath,'r')
    
    self.inlabels   = [x.decode('utf-8') for x in th5['label_i'][:]]
    self.xmin       = th5['xmin'][:]
    self.xmax       = th5['xmax'][:]
    self.wavelength = th5['wavelengths'][:]
    self.resolution = np.array(th5['resolution'],dtype=float)

    if kwargs.get('testing',False):
        self.testlabels = th5['testlabels'][:]
        self.testpred   = th5['testpred'][:]
        self.testmedflux = th5['testpred_medflux'][:]

    self.NNtype = kwargs.get('NNtype','LinNet')

    self.model = readNN(self.nnpath,NNtype=self.NNtype)

    th5.close()

  def eval(self,x):

    if isinstance(x,list):
        x = np.asarray(x)
    if len(x.shape) == 1:
        inputD = 1
    else:
        inputD = x.shape[0]

    inputVar = Variable(torch.from_numpy(x).type(dtype)).reshape(inputD,self.model.D_in)
    outmod = self.model(inputVar)
    outmod = outmod.data.numpy().squeeze()

    return outmod


class PayneContPredict(object):
     """
     Class for taking a Payne-learned NN and predicting continuum.
     """
     def __init__(self, nnpath=None, **kwargs):
          self.NN = {}
          if nnpath != None:
               self.nnpath = nnpath
          else:
               # define aliases for the MIST isochrones and C3K/CKC files
               self.nnpath  = Payne.__abspath__+'data/specANN/YSTANN.h5'

          self.NNtype = kwargs.get('NNtype','LinNet')
          self.anns = ANN(nnpath=self.nnpath,NNtype=self.NNtype,testing=False,verbose=False)

          # # check to see if using NN with Teff / 1000.0
          # if self.anns.xmin[0] < 1000.0:
          #      self.anns.xmin[0] = self.anns.xmin[0] * 1000.0
          #      self.anns.xmax[0] = self.anns.xmax[0] * 1000.0


     def predictcont(self,labels):
          '''
          predict continuum using set of labels and trained NN output

          :params labels:
          list of label values for the labels used to train the NN
          ex. [Teff,log(g),[Fe/H],[alpha/Fe]]

          :returns predict_flux:
          predicted flux from the NN, last element is the median flux
          '''

          predict_cont = self.anns.eval(labels)

          return predict_cont

     def getcont(self,**kwargs):
          '''
          function to take a set of kwarg based on labels and 
          return the predicted spectrum

          default returns solar continuum

          : returns modwave:
          Wavelength array from the NN

          :returns modspec:
          Predicted spectrum from the NN

          '''

          self.inputdict = {}

          if 'Teff' in kwargs:
               self.inputdict['teff'] = kwargs['Teff'] 
          elif 'logt' in kwargs:
               self.inputdict['teff'] = (10.0**kwargs['logt']) 
          else:
               self.inputdict['teff'] = 5770.0

          if 'log(g)' in kwargs:
               self.inputdict['logg'] = kwargs['log(g)']
          elif 'logg' in kwargs:
               self.inputdict['logg'] = kwargs['logg']
          else:
               self.inputdict['logg'] = 4.44

          if '[Fe/H]' in kwargs:
               self.inputdict['feh'] = kwargs['[Fe/H]']
          elif 'feh' in kwargs:
               self.inputdict['feh'] = kwargs['feh']
          else:
               self.inputdict['feh'] = 0.0

          if '[alpha/Fe]' in kwargs:
               self.inputdict['afe'] = kwargs['[alpha/Fe]']
          elif '[a/Fe]' in kwargs:
               self.inputdict['afe'] = kwargs['[a/Fe]']
          elif 'aFe' in kwargs:
               self.inputdict['afe'] = kwargs['aFe']
          elif 'afe' in kwargs:
               self.inputdict['afe'] = kwargs['afe']
          else:
               self.inputdict['afe'] = 0.0

          outwave = kwargs.get('outwave',None)

          # determine if NN has vmic built into it by seeing if kwargs['vmic'] == np.nan
          if 'vmic' in kwargs:
               if np.isfinite(kwargs['vmic']):
                    self.inputdict['vmic'] = kwargs['vmic']
                    usevmicbool = True
               else:
                    self.inputdict['vmic'] = np.nan
                    usevmicbool = False
          else:
               self.inputdict['vmic'] = np.nan
               usevmicbool = False

          # calculate model continuum at the native network resolution
          if usevmicbool:
               modcont = self.predictcont([self.inputdict[kk] for kk in ['teff','logg','feh','afe','vmic']])
          else:
               modcont = self.predictcont([self.inputdict[kk] for kk in ['teff','logg','feh','afe']])

          # pull out median flux from NN prediction
          modcont = modcont[:-1] * (10.0**modcont[-1])
          # modmedflux = 10.0**modcont[-1]

          # multiply cont by modflux
          # modcont = modcont * modmedflux

          modcontwave = self.anns.wavelength

          # convert the continuum from F_nu -> F_lambda
          # modcont *= (speedoflight/((modcontwave*1E-8)**2.0))

          inst_R_bool = False
          if 'inst_R' in kwargs:
               if outwave is not None:
                    outwave = np.array(outwave)

               if isinstance(kwargs['inst_R'],float):
                    # check to make sure inst_R != 0.0
                    if kwargs['inst_R'] > 0.0:
                         inst_R_bool = True

                         modcont = self.smoothspec(
                              modcontwave,modcont,kwargs['inst_R'],
                              outwave=outwave,smoothtype='R',
                              fftsmooth=True,inres=self.anns.resolution)

               else:
                    # LSF case where inst_R is vector of dispersion in AA at each pixel
                    inst_R_bool = True

                    # interpolate dispersion array onto input wave array
                    if outwave is not None:
                         disparr = np.interp(
                              modcontwave,outwave,kwargs['inst_R'])
                    else:
                         disparr = kwargs['inst_R']

                    # check to see if len(inst_R) == len(modwave)
                    try:
                         assert len(disparr) == len(modcontwave)
                    except AssertionError:
                         print('Length of LSF vector not equal to input wavelength')
                         raise
                    
                    modcont = self.smoothspec(
                         modcontwave,modcont,disparr,
                         outwave=outwave,smoothtype='lsf',
                         fftsmooth=True,inres=self.anns.resolution)

          if (inst_R_bool == False) & (outwave is not None):
               modcont = np.interp(kwargs['outwave'],modcontwave,modcont,right=np.nan,left=np.nan)

          if outwave is not None:
               modcontwave = outwave

          return modcontwave, modcont

     def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
          outspec = smoothspec(wave, spec, resolution=sigma, outwave=outwave, **kwargs)
          return outspec