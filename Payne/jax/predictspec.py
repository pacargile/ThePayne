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

class PayneSpecPredict(object):
     """
     Class for taking a Payne-learned NN and predicting spectrum.
     """
     def __init__(self, nnpath=None, **kwargs):
          self.NN = {}
          if nnpath != None:
               self.nnpath = nnpath
          else:
               # define aliases for the MIST isochrones and C3K/CKC files
               self.nnpath  = Payne.__abspath__+'data/specANN/YSTANN.h5'

          self.NNtype = kwargs.get('NNtype','LinNet')
          self.C_NNtype = kwargs.get('C_NNtype','LinNet')
          self.anns = ANN(nnpath=self.nnpath,NNtype=self.NNtype,testing=False,verbose=False)

          # self.anns = Net(self.nnpath)

          # # check to see if using NN with Teff / 1000.0
          # if self.anns.xmin[0] < 1000.0:
          #      self.anns.xmin.at[0].multiply(1000.0)
          #      self.anns.xmax.at[0].multiply(1000.0)

          # NNtype = kwargs.get('NNtype','YST1')
          # if NNtype == 'YST1':
          #      self.modpars = ['teff','logg','feh','afe']
          # else:
          #      self.modpars = ['teff','logg','feh','afe','vmic']

          self.modpars = self.anns.inlabels

          self.Cnnpath = kwargs.get('Cnnpath',None)
          if self.Cnnpath is not None:
               self.Canns_bool = True
               self.Canns = ANN(
                    nnpath=self.Cnnpath,
                    NNtype=self.C_NNtype,
                    testing=False,
                    verbose=False)
               self.contfn = self.predictcont
          else:
               self.Canns_bool = False
               self.Canns = None
               self.contfn = lambda _ : 1.0


     def predictspec(self,labels):
          '''
          predict spectra using set of labels and trained NN output

          :params labels:
          list of label values for the labels used to train the NN
          ex. [Teff,log(g),[Fe/H],[alpha/Fe]]

          :returns predict_flux:
          predicted flux from the NN
          '''

          self.predict_flux = self.anns.eval(labels)

          return self.predict_flux

     def predictcont(self,labels):
          '''
          predict continuum using set of labels and trained NN output

          :params labels:
          list of label values for the labels used to train the NN
          ex. [Teff,log(g),[Fe/H],[alpha/Fe]]

          :returns predict_flux:
          predicted flux from the NN
          '''

          predict_cont = self.Canns.eval(labels)
          modcontwave = self.Canns.wavelength

          # convert the continuum from F_nu -> F_lambda
          modcont = predict_cont * (speedoflight/((modcontwave*1E-8)**2.0))

          # normalize the continuum
          modcont = modcont / np.nanmedian(modcont)

          # interpolate continuum onto spectrum
          return np.interp(self.anns.wavelength,modcontwave,modcont,right=np.nan,left=np.nan)

     def getspec(self,**kwargs):
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

          modspec = self.predictspec([self.inputdict[kk] for kk in self.modpars])
          modwave = self.anns.wavelength

          modspec = modspec * self.contfn([self.inputdict[kk] for kk in self.modpars])

          rot_vel_bool = False
          if 'rot_vel' in kwargs:
               # check to make sure rot_vel isn't 0.0, this will cause the convol. to crash
               # if kwargs['rot_vel'] != 0.0:
               # set boolean to let rest of code know the spectrum has been broadened
               rot_vel_bool = True

               # use B.Johnson's smoothspec to convolve with rotational broadening
               modspec = self.smoothspec(modwave,modspec,kwargs['rot_vel'],
                    outwave=None,smoothtype='vsini',fftsmooth=True,inres=0.0)
               modspec = modspec.at[0].set(modspec[1])
               modspec = modspec.at[-1].set(modspec[-2])

          rad_vel_bool = False
          if 'rad_vel' in kwargs:
               # if kwargs['rad_vel'] != 0.0:
               #      # kwargs['radial_velocity']: RV in km/s
               rad_vel_bool = True
               # modwave = self.NN['wavelength'].copy()*(1.0-(kwargs['rad_vel']/speedoflight))
               modwave = modwave*(1.0+(kwargs['rad_vel']/speedoflight))


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
                         outwave = modwave
                    else:
                         outwave = np.array(kwargs['outwave'])
               else:
                    outwave = modwave

               if np.iterable(kwargs['inst_R']):
                    smoothtype = 'lsf'
                    lsf = np.interp(modwave,outwave,kwargs['inst_R'])
               else:
                    smoothtype = 'R'
                    lsf = 2.355*kwargs['inst_R']
               modspec = self.smoothspec(modwave,modspec,lsf,
                    outwave=outwave,smoothtype=smoothtype,fftsmooth=True,
                    inres=self.anns.resolution)
               modspec = modspec.at[0].set(modspec[1])
               modspec = modspec.at[-1].set(modspec[-2])

               if outwave is not None:
                    modwave = outwave

          # if kwargs['outwave'] is not None:
          #      modspec = np.interp(kwargs['outwave'],modwave,modspec,right=np.nan,left=np.nan)

          if (inst_R_bool == False) & ('outwave' in kwargs):
               if kwargs['outwave'] is not None:
                    modspec = np.interp(kwargs['outwave'],modwave,modspec,right=np.nan,left=np.nan)

          return modwave, modspec

     def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
          outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
          return outspec