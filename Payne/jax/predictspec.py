# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import jax.numpy as np
import jax.scipy as jsp
from jax import jit,vmap,lax
from jax import dtypes

import warnings
from datetime import datetime
import h5py
import numpy as onp
from scipy import constants
speedoflight = constants.c / 1000.0
from functools import partial

import Payne

# import torch
# from torch import nn
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if str(device) != 'cpu':
#   dtype = torch.cuda.FloatTensor
# else:
#   dtype = torch.FloatTensor
# import torch.nn.functional as F

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
          self.anns = ANN(nnpath=self.nnpath,NNtype=self.NNtype,testing=False,verbose=False)

          # labels for the spectral NN
          self.modpars = self.anns.inlabels

          # rename vturb to vmic if it's in the modpars, since some NNs use vturb and some use vmic
          if 'vturb' in self.modpars:
               self.modpars = ['vmic' if kk == 'vturb' else kk for kk in self.modpars]

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
          return np.interp(self.anns.wavelength,modcontwave,modcont)

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
          elif 'teff' in kwargs:
               self.inputdict['teff'] = kwargs['teff'] 
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
               self.inputdict['vmic'] = kwargs['vmic']
          elif 'vturb' in kwargs:
               self.inputdict['vmic'] = kwargs['vturb']
          else:
               self.inputdict['vmic'] = 1.0

          if 'av' in kwargs:
               self.inputdict['av'] = kwargs['av']
          else:
               self.inputdict['av'] = 0.0
          
          if 'rv' in kwargs:
               self.inputdict['rv'] = kwargs['rv']
          else:
               self.inputdict['rv'] = 3.1

          modspec = self.predictspec([self.inputdict[kk] for kk in self.modpars])
          modwave = self.anns.wavelength

          modspec = modspec * self.contfn([self.inputdict[kk] for kk in self.Cmodpars])

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

class PayneSpecPredictNew(object):
     def __init__(self, nnpath=None, nntype='MLP_v1', norm=False):
          self.nnpath = nnpath
          self.nntype = nntype
          self.norm = norm
        
          from .specNN import modpred
          self.anns = modpred(nnpath=self.nnpath,nntype=self.nntype,norm=self.norm)
          
          # labels for the spectral NN
          self.modpars = self.anns.inlabels

          # Build a jitted “core” that closes over self.anns
          self._spec_core = self._build_spec_core(self.anns)
     
     def predictspec(self,labels):
          '''
          predict spectra using set of labels and trained NN output

          :params labels:
          list of label values for the labels used to train the NN
          ex. [Teff,log(g),[Fe/H],[alpha/Fe]]

          :returns predict_flux:
          predicted flux from the NN
          '''

          self.predict_flux = self.anns.predspec(labels)

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

          self.verbose = kwargs.get('verbose',False)

          self.inputdict = {}

          if 'Teff' in kwargs:
               self.inputdict['teff'] = kwargs['Teff'] 
          elif 'teff' in kwargs:
               self.inputdict['teff'] = kwargs['teff']
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
               self.inputdict['vmic'] = kwargs['vmic']
          elif 'vturb' in kwargs:
               self.inputdict['vmic'] = kwargs['vturb']
          else:
               self.inputdict['vmic'] = 1.0

          if 'vstar' in kwargs:
               self.inputdict['rot_vel'] = kwargs['vstar']
          elif 'rot_vel' in kwargs:
               self.inputdict['rot_vel'] = kwargs['rot_vel']
          else:
               self.inputdict['rot_vel'] = 0.0

          if 'vrad' in kwargs:
               self.inputdict['rad_vel'] = kwargs['vrad']
          elif 'rad_vel' in kwargs:
               self.inputdict['rad_vel'] = kwargs['rad_vel']
          else:
               self.inputdict['rad_vel'] = 0.0

          if 'Av' in kwargs:
               self.inputdict['av'] = kwargs['Av']
          elif 'av' in kwargs:
               self.inputdict['av'] = kwargs['av']
          else:
               self.inputdict['av'] = 0.0

          if 'Rv' in kwargs:
               self.inputdict['rv'] = kwargs['Rv']
          elif 'rv' in kwargs:
               self.inputdict['rv'] = kwargs['rv']
          else:
               self.inputdict['rv'] = 3.1
    
          if 'inst_R' in kwargs:
               self.inputdict['inst_R'] = kwargs['inst_R']
          else:
               self.inputdict['inst_R'] = 0.0
    
          outwave = kwargs.get('outwave', None)
    
          pars = [
               self.inputdict['teff'],
               self.inputdict['logg'],
               self.inputdict['feh'],
               self.inputdict['afe'],
               self.inputdict['vmic'],
               self.inputdict['av'],
               self.inputdict['rv'],
               ]
          pars = np.asarray(pars, dtype=np.float32)
          
          if self.verbose:
               print('Input labels for spectrum prediction:')
               for kk in self.inputdict.keys():
                    if kk != 'inst_R':
                         print('  {0} = {1}'.format(kk,self.inputdict[kk]))

          apply_rot  = self._to_bool_flag(self.inputdict['rot_vel'])
          apply_rv   = self._to_bool_flag(self.inputdict['rad_vel'])
          apply_inst = self._to_bool_flag(self.inputdict['inst_R'])

          modwave, modspec = self._spec_core(
               pars,
               self.inputdict['rot_vel'],
               self.inputdict['rad_vel'],
               self.inputdict['inst_R'],
               outwave,
               apply_rot,
               apply_rv,
               apply_inst,
          )
          
          return modwave, modspec

     def _to_bool_flag(self, x):
          """
          Convert x (Python/NumPy/JAX scalar or array) into a Python bool
          indicating whether any element is non-zero.
          """
          x_np = onp.asarray(x)   # moves JAX value to host
          return bool(onp.any(x_np != 0.0))


     def _build_spec_core(self, anns):
          """
          Create a jitted core that knows how to call anns.predspec.
          `anns` is captured in the closure at construction time.
          """
          anns_resolution = anns.resolution
          modwave0 = anns.wavelength          

          def core(pars, rot_vel, rad_vel, inst_R, outwave,
               apply_rot: bool, apply_rv: bool, apply_inst: bool):
               # 1) intrinsic spectrum from the NN
               modspec = anns.predspec(pars)      
               modwave = modwave0

               # 2) rotational broadening
               if apply_rot:
                    modspec = lax.cond(
                         rot_vel > 0.0,
                         lambda s: smoothspec(modwave, s, rot_vel, outwave=None,
                                             smoothtype="vsini", fftsmooth=True, inres=0.0),
                         lambda s: s,
                         modspec,
                    )
                    modspec = modspec.at[0].set(modspec[1])
                    modspec = modspec.at[-1].set(modspec[-2])

               # 3) RV shift
               if apply_rv:
                    c = 2.99792458e5
                    modwave = lax.cond(
                         rad_vel != 0.0,
                         lambda w: w * (1.0 + rad_vel / c),
                         lambda w: w,
                         modwave,
                    )

               # 4) instrumental broadening
               if apply_inst:
                    def _do_inst(args):
                         modwave, modspec, inst_R, outwave = args
                         if outwave is None:
                              outwave = modwave

                         # here you decide scalar vs array inst_R via shapes, not np.iterable
                         def branch_lsf(args2):
                              mw, ms, inst_R, ow = args2
                              # choose wave grid on which inst_R is defined
                              base_wave = ow if ow is not None else mw

                              # make sure inst_R is defined on base_wave
                              inst_R_arr = np.asarray(inst_R)

                              if inst_R_arr.ndim == 0:
                                   # scalar R → treat as constant resolution over wavelength
                                   inst_R_arr = np.full_like(base_wave, inst_R_arr)
                              elif inst_R_arr.shape != base_wave.shape:
                                   # assume inst_R is defined on mw; interpolate to base_wave
                                   # (now both xp and fp are 1-D with equal length)
                                   inst_R_arr = np.interp(base_wave, mw, inst_R_arr)

                              # intrinsic resolution of the NN: anns_resolution is Rsigma_in (lambda/sigma)
                              # convert to sigma_lambda for input and output
                              sigma_in = base_wave / anns_resolution         # sigma_in(λ)
                              sigma_out = base_wave / inst_R_arr             # sigma_out(λ)

                              # kernel sigma: subtract in quadrature, clip to >=0
                              sigma_k_sq = sigma_out**2 - sigma_in**2
                              sigma_k_sq = np.clip(sigma_k_sq, 0.0, np.inf)
                              sigma_k = np.sqrt(sigma_k_sq)                  # sigma_k(λ)

                              # apply wavelength-dependent smoothing as an LSF
                              return smoothspec(
                                   mw,
                                   ms,
                                   sigma_k,
                                   outwave=ow,
                                   smoothtype="lsf",
                                   fftsmooth=True,
                              )
    
                         def branch_R(args2):
                              mw, ms, inst_R, ow = args2
                              # Ensure we always have a *scalar* R here, even if inst_R is an array
                              inst_R0 = np.reshape(inst_R, (-1,))[0]  # works for scalar or vector
                              lsf = 2.355 * inst_R0

                              return smoothspec(
                                   mw,
                                   ms,
                                   lsf,          # scalar resolution
                                   outwave=ow,
                                   smoothtype="R",
                                   fftsmooth=True,
                                   inres=anns_resolution,
                              )

                         # you can use lax.cond on (jnp.ndim(inst_R) > 0), etc.
                         outspec = lax.cond(
                              np.ndim(inst_R) > 0,
                              branch_lsf,
                              branch_R,
                              (modwave, modspec, inst_R, outwave),
                         )
                         outspec = outspec.at[0].set(outspec[1])
                         outspec = outspec.at[-1].set(outspec[-2])
                         return outspec

                    modspec = _do_inst((modwave, modspec, inst_R, outwave))

               return modwave, modspec

          # Booleans can be static so JAX doesn’t recompile for data changes
          return jit(core, static_argnames=('apply_rot', 'apply_rv', 'apply_inst'))

     def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
          outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
          return outspec


"""

          # modspec = self.predictspec(pars)
          # modwave = self.anns.wavelength

          # if self.verbose:
          #      print(f'Min/Max predicted flux: {np.min(modspec)} / {np.max(modspec)}')

          # rot_vel_bool = False
          # if ('rot_vel' in kwargs) and (kwargs['rot_vel'] > 0.0):
          #      # check to make sure rot_vel isn't 0.0, this will cause the convol. to crash
          #      # if kwargs['rot_vel'] != 0.0:
          #      # set boolean to let rest of code know the spectrum has been broadened
          #      rot_vel_bool = True
          #      if self.verbose:
          #           print('  Applying rotational broadening: vstar = {0} km/s'.format(kwargs['rot_vel']))

          #      # use B.Johnson's smoothspec to convolve with rotational broadening
          #      modspec = self.smoothspec(modwave,modspec,kwargs['rot_vel'],
          #           outwave=None,smoothtype='vsini',fftsmooth=True,inres=0.0)
          #      modspec = modspec.at[0].set(modspec[1])
          #      modspec = modspec.at[-1].set(modspec[-2])

          # rad_vel_bool = False
          # if ('rad_vel' in kwargs) and (kwargs['rad_vel'] != 0.0):
          #      # if kwargs['rad_vel'] != 0.0:
          #      #      # kwargs['radial_velocity']: RV in km/s
          #      rad_vel_bool = True
          #      if self.verbose:
          #           print('  Applying radial velocity shift: vrad = {0} km/s'.format(kwargs['rad_vel']))
          #      modwave = modwave*(1.0+(kwargs['rad_vel']/speedoflight))


          # inst_R_bool = False
          # if ('inst_R' in kwargs) and (kwargs['inst_R'] is not None):
          #      # check to make sure inst_R != 0.0
          #      # if kwargs['inst_R'] != 0.0:
          #      inst_R_bool = True
          #      # instrumental broadening
          #      # if rot_vel_bool:
          #      #     inres = (2.998e5)/kwargs['rot_vel']
          #      # else:
          #      #     inres = self.NN['resolution']
          #      # inres=None
          #      if self.verbose:
          #           print('  Applying instrumental broadening: R = {0}'.format(kwargs['inst_R']))

          #      if 'outwave' in kwargs:
          #           if kwargs['outwave'] is None:
          #                outwave = modwave
          #           else:
          #                outwave = np.array(kwargs['outwave'])
          #      else:
          #           outwave = modwave

          #      if np.iterable(kwargs['inst_R']):
          #           smoothtype = 'lsf'
          #           lsf = np.interp(modwave,outwave,kwargs['inst_R'])
          #      else:
          #           smoothtype = 'R'
          #           lsf = 2.355*kwargs['inst_R']
          #      modspec = self.smoothspec(modwave,modspec,lsf,
          #           outwave=outwave,smoothtype=smoothtype,fftsmooth=True,
          #           inres=self.anns.resolution)
          #      modspec = modspec.at[0].set(modspec[1])
          #      modspec = modspec.at[-1].set(modspec[-2])

          #      if outwave is not None:
          #           modwave = outwave

          # # if kwargs['outwave'] is not None:
          # #      modspec = np.interp(kwargs['outwave'],modwave,modspec,right=np.nan,left=np.nan)

          # if (inst_R_bool == False) & ('outwave' in kwargs):
          #      if kwargs['outwave'] is not None:
          #           modspec = np.interp(kwargs['outwave'],modwave,modspec,right=np.nan,left=np.nan)

"""