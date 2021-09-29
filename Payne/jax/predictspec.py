# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import jax.numpy as np
import jax.scipy as jsp
from jax.ops import index, index_add, index_update
from jax import jit,vmap
import warnings
from datetime import datetime
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import h5py
from scipy import constants
speedoflight = constants.c / 1000.0

import Payne

from Payne.jax.smoothing import smoothspec

class Net(object):
     def __init__(self, NNpath):
          self.readNN(nnpath=NNpath)

     def readNN(self,nnpath=''):

          th5 = h5py.File(nnpath,'r')
          self.w_array_0 = np.array(th5['w_array_0'],dtype=np.float32)
          self.w_array_1 = np.array(th5['w_array_1'],dtype=np.float32)
          self.w_array_2 = np.array(th5['w_array_2'],dtype=np.float32)
          self.b_array_0 = np.array(th5['b_array_0'],dtype=np.float32)
          self.b_array_1 = np.array(th5['b_array_1'],dtype=np.float32)
          self.b_array_2 = np.array(th5['b_array_2'],dtype=np.float32)
          self.xmin      = np.array(th5['x_min'],dtype=np.float32)
          self.xmax      = np.array(th5['x_max'],dtype=np.float32)

          self.wavelength = np.asarray(th5['wavelength'],dtype=np.float32)

          self.resolution = np.array(th5['resolution'],dtype=np.float32)[0]

          th5.close()



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

class PayneSpecPredict(object):
     """
     Class for taking a Payne-learned NN and predicting spectrum.
     """
     def __init__(self, nnpath, **kwargs):
          self.NN = {}
          if nnpath != None:
               self.nnpath = nnpath
          else:
               # define aliases for the MIST isochrones and C3K/CKC files
               self.nnpath  = Payne.__abspath__+'data/specANN/YSTANN.h5'

          self.anns = Net(self.nnpath)

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
               self.inputdict['teff'] = kwargs['Teff'] / 1000.0
          elif 'logt' in kwargs:
               self.inputdict['teff'] = (10.0**kwargs['logt']) / 1000.0
          else:
               self.inputdict['teff'] = 5770.0/1000.0

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

          # # determine if NN has vmic built into it by seeing if kwargs['vmic'] == np.nan
          # if 'vmic' in kwargs:
          #      if np.isfinite(kwargs['vmic']):
          #           self.inputdict['vmic'] = kwargs['vmic']
          #           usevmicbool = True
          #      else:
          #           self.inputdict['vmic'] = np.nan
          #           usevmicbool = False
          # else:
          #      self.inputdict['vmic'] = np.nan
          #    usevmicbool = False

          usevmicbool = False

          # calculate model spectrum at the native network resolution
          if usevmicbool:
               modspec = self.predictspec([self.inputdict[kk] for kk in ['teff','logg','feh','afe','vmic']])
          else:
               modspec = self.predictspec([self.inputdict[kk] for kk in ['teff','logg','feh','afe']])

          modwave = self.anns.wavelength

          rot_vel_bool = False
          if 'rot_vel' in kwargs:
               # check to make sure rot_vel isn't 0.0, this will cause the convol. to crash
               # if kwargs['rot_vel'] != 0.0:
               # set boolean to let rest of code know the spectrum has been broadened
               rot_vel_bool = True

               # use B.Johnson's smoothspec to convolve with rotational broadening
               modspec = self.smoothspec(modwave,modspec,kwargs['rot_vel'],
                    outwave=None,smoothtype='vsini',fftsmooth=True,inres=0.0)
               modspec = index_update(modspec, index[0], modspec[1])

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
                         outwave = None
                    else:
                         outwave = np.array(kwargs['outwave'])
               else:
                    outwave = None

               modspec = self.smoothspec(modwave,modspec,kwargs['inst_R'],
                    outwave=kwargs['outwave'],smoothtype='R',fftsmooth=True,
                    inres=self.anns.resolution)
               modspec = index_update(modspec, index[0], modspec[1])

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