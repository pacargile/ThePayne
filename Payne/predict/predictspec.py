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

    self.normed = kwargs.get('normed',False)

    if self.verbose:
      print('... Reading in {0}'.format(self.nnpath))
    th5 = h5py.File(self.nnpath,'r')
    
    if self.normed:
        self.ymin = th5['ymin'][()]
        self.ymax = th5['ymax'][()]

    self.xmin       = th5['xmin'][:]
    self.xmax       = th5['xmax'][:]
    self.wavelength = th5['wavelengths'][:]
    self.resolution = np.array(th5['resolution'],dtype=float)

    self.NNtype = kwargs.get('NNtype','LinNet')

    self.model = readNN(self.nnpath,NNtype=self.NNtype)

    th5.close()

  def eval(self,x):
    # check if Teff (always x[0]) is log'ed or not
    if x[0] > 100.0:
        x[0] = np.log10(x[0])

    if isinstance(x,list):
        x = np.asarray(x)
    if len(x.shape) == 1:
        inputD = 1
    else:
        inputD = x.shape[0]

    inputVar = Variable(torch.from_numpy(x).type(dtype)).reshape(inputD,self.model.D_in)
    outmod = self.model(inputVar)
    outmod = outmod.data.numpy().squeeze()

    # outpars = self.model.npeval(x)

    if self.normed:
        outmod = np.array([self.unnorm(x,ii) for ii,x in enumerate(outmod)])
    return outmod

  def unnorm(self,x,ii):
    return (x + 0.5)*(self.ymax[ii]-self.ymin[ii]) + self.ymin[ii]


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

          self.NNtype = kwargs.get('NNtype','SMLP')
          self.C_NNtype = kwargs.get('C_NNtype','SMLP')
          self.anns = readNN(self.nnpath,NNtype=self.NNtype)

          # # check to see if using NN with Teff / 1000.0
          # if self.anns.xmin[0] < 1000.0:
          #      self.anns.xmin[0] = self.anns.xmin[0] * 1000.0
          #      self.anns.xmax[0] = self.anns.xmax[0] * 1000.0

          self.Cnnpath = kwargs.get('Cnnpath',None)
          if self.Cnnpath is not None:
               self.Canns = readNN(self.Cnnpath,NNtype=self.C_NNtype)
          else:
               self.Canns = None

     def predictspec(self,labels):
          '''
          predict spectra using set of labels and trained NN output

          :params labels:
          list of label values for the labels used to train the NN
          ex. [Teff,log(g),[Fe/H],[alpha/Fe]]

          :returns predict_flux:
          predicted flux from the NN
          '''

          predict_flux = self.anns.eval(labels)

          return predict_flux

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

          return predict_cont

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

          # calculate model spectrum at the native network resolution
          if usevmicbool:
               modspec = self.predictspec([self.inputdict[kk] for kk in ['teff','logg','feh','afe','vmic']])
          else:
               modspec = self.predictspec([self.inputdict[kk] for kk in ['teff','logg','feh','afe']])

          modwave = self.anns.wavelength

          if self.Canns is not None:
               # calculate model continuum at the native network resolution
               if usevmicbool:
                    modcont = self.predictcont([self.inputdict[kk] for kk in ['teff','logg','feh','afe','vmic']])
               else:
                    modcont = self.predictcont([self.inputdict[kk] for kk in ['teff','logg','feh','afe']])

               modcontwave = self.Canns.wavelength

               # convert the continuum from F_nu -> F_lambda
               modcont = modcont * (speedoflight/((modcontwave*1E-8)**2.0))

               # normalize the continuum
               modcont = modcont / np.nanmedian(modcont)

               # interpolate continuum onto spectrum
               modspec = modspec * np.interp(
                    modwave,modcontwave,modcont,
                    right=np.nan,left=np.nan)

          rot_vel_bool = False
          if 'rot_vel' in kwargs:
               # check to make sure rot_vel isn't 0.0, this will cause the convol. to crash
               if kwargs['rot_vel'] != 0.0:
                    # set boolean to let rest of code know the spectrum has been broadened
                    rot_vel_bool = True

                    # use B.Johnson's smoothspec to convolve with rotational broadening
                    modspec = self.smoothspec(modwave,modspec,
                         kwargs['rot_vel'],
                         outwave=None,smoothtype='vsini',
                         fftsmooth=True,inres=0.0)
                    modspec[0] = modspec[1]
                    modspec[-1] = modspec[-2]

          rad_vel_bool = False
          if 'rad_vel' in kwargs:
               if kwargs['rad_vel'] != 0.0:
                    # kwargs['radial_velocity']: RV in km/s
                    rad_vel_bool = True
                    # modwave = self.NN['wavelength'].copy()*(1.0-(kwargs['rad_vel']/speedoflight))
                    modwave = modwave*(1.0+(kwargs['rad_vel']/speedoflight))
          inst_R_bool = False
          if 'inst_R' in kwargs:
               if outwave is not None:
                    outwave = np.array(outwave)

               if isinstance(kwargs['inst_R'],float):
                    # check to make sure inst_R != 0.0
                    if kwargs['inst_R'] > 0.0:
                         inst_R_bool = True

                         modspec = self.smoothspec(
                              modwave,modspec,kwargs['inst_R'],
                              outwave=outwave,smoothtype='R',
                              fftsmooth=True,inres=self.anns.resolution)

               else:
                    # LSF case where inst_R is vector of dispersion in AA at each pixel
                    inst_R_bool = True

                    # interpolate dispersion array onto input wave array
                    if outwave is not None:
                         disparr = np.interp(
                              modwave,outwave,kwargs['inst_R'])
                    else:
                         disparr = kwargs['inst_R']

                    # check to see if len(inst_R) == len(modwave)
                    try:
                         assert len(disparr) == len(modwave)
                    except AssertionError:
                         print('Length of LSF vector not equal to input wavelength')
                         raise
                    
                    modspec = self.smoothspec(
                         modwave,modspec,disparr,
                         outwave=outwave,smoothtype='lsf',
                         fftsmooth=True,inres=self.anns.resolution)

          if (inst_R_bool == False) & (outwave is not None):
               modspec = np.interp(kwargs['outwave'],modwave,modspec,right=np.nan,left=np.nan)

          if outwave is not None:
               modwave = outwave

          return modwave, modspec

     def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
          outspec = smoothspec(wave, spec, resolution=sigma, outwave=outwave, **kwargs)
          return outspec