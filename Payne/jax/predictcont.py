# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import jax.numpy as np
import jax.scipy as jsp
from jax import jit,vmap,lax
import warnings
from datetime import datetime
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import h5py
from scipy import constants
speedoflight = constants.c / 1000.0

import Payne

import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != 'cpu':
  dtype = torch.cuda.FloatTensor
else:
  dtype = torch.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F

from Payne.jax.smoothing import smoothspec
from Payne.jax.NNmodels import readNN, YSTNet

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

          try:
               self.inlabels   = [x.decode('utf-8') for x in th5['label_i'][:]]
               self.wavelength = th5['wavelengths'][:]
               self.resolution = np.array(th5['resolution'],dtype=float)
          except:
               self.inlabels = ['teff','logg','feh','afe']
               self.wavelength = th5['wavelength'][:]
               self.resolution = np.array(th5['resolution'],dtype=float)

          if kwargs.get('testing',False):
               self.testlabels = th5['testlabels'][:]
               self.testpred   = th5['testpred'][:]

          self.NNtype = kwargs.get('NNtype','LinNet')

          if self.NNtype != 'YST1':
               self.model = readNN(self.nnpath,NNtype=self.NNtype)
          else:
               self.model = YSTNet(self.nnpath)

          th5.close()

     def eval(self,x):

          if isinstance(x,list):
               x = np.asarray(x)
          if len(x.shape) == 1:
               inputD = 1
          else:
               inputD = x.shape[0]

          inputVar = x
          outmod = self.model.npeval(inputVar)

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

          # labels for the spectral NN
          self.modpars = self.anns.inlabels

          self.Cnnpath = kwargs.get('Cnnpath',None)
          self.C_NNtype = kwargs.get('C_NNtype','LinNet')
          if self.Cnnpath is not None:
               self.Canns_bool = True
               self.Canns = ANN(
                    nnpath=self.Cnnpath,
                    NNtype=self.C_NNtype,
                    testing=False,
                    verbose=False)
               self.contfn = self.predictcont
               self.Cmodpars = self.Canns.inlabels
          else:
               self.Canns_bool = False
               self.Canns = None
               self.contfn = lambda _ : 1.0
               self.Cmodpars = []


     def predictcont(self,labels):
          '''
          predict continuum using set of labels and trained NN output

          :params labels:
          list of label values for the labels used to train the NN
          ex. [Teff,log(g),[Fe/H],[alpha/Fe]]

          :returns predict_flux:
          predicted flux from the NN
          '''

          self.predict_flux = self.anns.eval(labels)

          return self.predict_flux

     def getcont(self,**kwargs):
          '''
          function to take a set of kwarg based on labels and 
          return the predicted spectrum

          default returns solar spectrum, rotating at 2 km/s, and 
          at R=32K

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

          if 'vmic' in kwargs:
               self.inputdict['vturb'] = kwargs['vmic']
          elif 'vtrub' in kwargs:
               self.inputdict['vtrub'] = kwargs['vturb']
          else:
               self.inputdict['vturb'] = np.nan


          modcont = self.predictcont([self.inputdict[kk] for kk in self.modpars])

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
               # check to make sure inst_R != 0.0
               # if kwargs['inst_R'] != 0.0:
               inst_R_bool = True
               # instrumental broadening
               # if rot_vel_bool:
               #     inres = (2.998e5)/kwargs['rot_vel']
               # else:
               #     inres = self.NN['resolution']
               # inres=None
               if 'outwave' in kwargs:
                    if kwargs['outwave'] is None:
                         outwave = modcontwave
                    else:
                         outwave = np.array(kwargs['outwave'])
               else:
                    outwave = modcontwave

               if np.iterable(kwargs['inst_R']):
                    smoothtype = 'lsf'
                    lsf = np.interp(modcontwave,outwave,kwargs['inst_R'])
               else:
                    smoothtype = 'R'
                    lsf = 2.355*kwargs['inst_R']
               modcont = self.smoothspec(modcontwave,modcont,lsf,
                    outwave=outwave,smoothtype=smoothtype,fftsmooth=True,
                    inres=self.anns.resolution)
               modcont = modcont.at[0].set( modcont[1])
               modcont = modcont.at[-1].set(modcont[-2])

               if outwave is not None:
                    modcontwave = outwave

          # if kwargs['outwave'] is not None:
          #      modspec = np.interp(kwargs['outwave'],modwave,modspec,right=np.nan,left=np.nan)

          if (inst_R_bool == False) & ('outwave' in kwargs):
               if kwargs['outwave'] is not None:
                    modcont = np.interp(kwargs['outwave'],modcontwave,modcont,right=np.nan,left=np.nan)

          return modcontwave, modcont

     def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
          outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
          return outspec