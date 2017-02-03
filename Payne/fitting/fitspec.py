# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from scipy import constants
speedoflight = constants.c / 1000.0
from scipy.interpolate import UnivariateSpline

from ..predict.predictspec import PaynePredict
from .fitters import SpecMinimize,SpecEmcee

class FitSpec(object):
	"""docstring for FitSpec"""
	def __init__(self, NN):
		# user inputed neural-net output file
		self.NN = NN

		# initialize the Payne Predictor
		self.PP = PaynePredict(self.NN)


	def run(self,obsdict,pinit,method='minimize'):
		# initialize a dict for input arguments into the fitter
		self.fitargs = {}
		self.fitargs['obs_wave'] = obsdict['obs_wave']
		self.fitargs['obs_flux'] = obsdict['obs_flux']
		self.fitargs['obs_eflux'] = obsdict['obs_eflux']
		self.fitargs['inst_R'] = obsdict['inst_R']
		if 'wave_minmax' in obsdict.keys():
			self.fitargs['wave_minmax'] = obsdict['wave_minmax']
			wavecond = ((self.fitargs['obs_wave'] >= obsdict['wave_minmax'][0]) 
				& (self.fitargs['obs_wave'] >= obsdict['wave_minmax'][1]))
			self.fitargs['obs_wave_fit']  = self.fitargs['obs_wave'][wavecond]
			self.fitargs['obs_flux_fit']  = self.fitargs['obs_flux'][wavecond]
			self.fitargs['obs_eflux_fit'] = self.fitargs['obs_eflux'][wavecond]
		else:
			self.fitargs['obs_wave_fit']  = self.fitargs['obs_wave']
			self.fitargs['obs_flux_fit']  = self.fitargs['obs_flux']
			self.fitargs['obs_eflux_fit'] = self.fitargs['obs_eflux']


		if method == 'minimize':
			self.fitter = SpecMinimize(self.chi2func)
			self.pars = self.fitter.run(pinit)

			return self.pars

	def chi2func(self,pars,*args,**kwargs):
		Teff = pars[0]
		logg = pars[1]
		FeH  = pars[2]
		radvel = pars[3]
		rotvel = pars[4]

		# check to make sure pars are in grid used to train NN
		if (Teff < self.PP.NN['xmin'][0]) | (Teff > self.PP.NN['xmax'][0]):
			return np.nan_to_num(np.inf)
		if (logg < self.PP.NN['xmin'][1]) | (logg > self.PP.NN['xmax'][1]):
			return np.nan_to_num(np.inf)
		if (FeH < self.PP.NN['xmin'][2])  | (FeH > self.PP.NN['xmax'][2]):
			return np.nan_to_num(np.inf)

		if (np.abs(radvel) > 1000.0):
			return np.nan_to_num(np.inf)
		if (rotvel <= 0) | (rotvel > 1.0E+3):
			return np.nan_to_num(np.inf)

		# predict model flux at model wavelengths
		modwave_i,modflux_i = self.PP.getspec(
			Teff=Teff,logg=logg,feh=FeH,rad_vel=radvel,rot_vel=rotvel,inst_R=self.fitargs['inst_R'])

		# linearly interpolate model fluxes onto observed wavelengths
		modflux = UnivariateSpline(modwave_i,modflux_i,k=1,s=0)(self.fitargs['obs_wave_fit'])

		# calc chi-square
		chi2 = np.sum( 
			[((m-o)**2.0)/(s**2.0) for m,o,s in zip(
				modflux,self.fitargs['obs_flux_fit'],self.fitargs['obs_eflux_fit'])])

		return chi2

	def lnprob(self):
		pass

	def lnlike(self):
		pass

	def lnprior(self):
		pass
