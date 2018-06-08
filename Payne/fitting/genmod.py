# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime

from .fitutils import polycalc

class GenMod(object):
	"""docstring for GenMod"""
	def __init__(self, *arg, **kwargs):
		super(GenMod, self).__init__()
		self.verbose = kwargs.get('verbose',False)
		
	def _initspecnn(self,nnpath=None,oldnn=False):
		if oldnn == False:
			from ..predict.predictspec import PayneSpecPredict
		else:
			from ..predict.oldpredictspec import PaynePredict

		from .fitutils import polycalc

		# initialize the Payne Spectrum Predictor
		self.PP = PayneSpecPredict(nnpath)

	def _initphotnn(self,filterarray,nnpath=None):
		self.filterarray = filterarray

		from ..predict.predictsed import FastPayneSEDPredict
		self.fppsed = FastPayneSEDPredict(
			usebands=self.filterarray,nnpath=nnpath,
			)

		# from ..predict.photANN import ANN


		# # for each input filter, read in the ANN file
		# ANNdict = {}
		# for ff in self.filterarray:
		# 	try:
		# 		ANNdict[ff] = ANN(ff,nnpath=nnpath,verbose=self.verbose)
		# 	except IOError:
		# 		print('Cannot find NN HDF5 file for {0}'.format(ff))
		# self.ANNdict = ANNdict

	def genspec(self,pars,outwave=None,verbose=False,normspec_bool=False):
		# define parameters from pars array
		Teff = pars[0]
		logg = pars[1]
		FeH  = pars[2]
		aFe = pars[3]
		radvel = pars[4]
		rotvel = pars[5]
		inst_R = pars[6]

		# check to see if a polynomial is used for spectrum normalization
		if normspec_bool:
			polycoef = pars[7:]
		# predict model flux at model wavelengths
		modwave_i,modflux_i = self.PP.getspec(
			Teff=Teff,logg=logg,feh=FeH,afe=aFe,rad_vel=radvel,rot_vel=rotvel,inst_R=2.355*inst_R,
			outwave=outwave)		
		# if polynomial normalization is turned on then multiply model by it
		if normspec_bool:
			epoly = polycalc(polycoef,outwave)
			# now multiply the model by the polynomial normalization poly
			modflux_i = modflux_i*epoly

		return modwave_i,modflux_i

	def genphot(self,pars,verbose=False):
		# define parameters from pars array
		Teff = pars[0]
		logg = pars[1]
		FeH  = pars[2]
		logR = pars[3]
		Dist = pars[4]
		Av   = pars[5]

		logTeff = np.log10(Teff)

		logL = 2.0*logR + 4.0*(logTeff - np.log10(5770.0))

		# create parameter dictionary
		photpars = {}
		photpars['logt'] = logTeff
		photpars['logg'] = logg
		photpars['feh']  = FeH
		photpars['logl'] = logL
		photpars['av']   = Av
		photpars['dist'] = Dist

		# create filter list and arrange photometry to this list

		# sed = self.ppsed.sed(filters=filterlist,**photpars)
		sed = self.fppsed.sed(**photpars)

		outdict = {ff_i:sed_i for sed_i,ff_i in zip(sed,self.filterarray)}

		# # calculate absolute bolometric magnitude
		# Mbol = -2.5*(2.0*logR + 4.0*np.log10(Teff/5770.0)) + 4.74

		# # calculate BC for all photometry in obs_phot dict
		# outdict = {}
		# for ii,kk in enumerate(self.filterarray):
		# 	BCdict_i = float(self.ANNdict[kk].eval([Teff,logg,FeH,Av]))
		# 	outdict[kk] = (Mbol - BCdict_i) + 5.0*np.log10(Dist) - 5.0

		return outdict
