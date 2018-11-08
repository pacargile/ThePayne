import numpy as np
from .genmod import GenMod
from datetime import datetime

class likelihood(object):
	"""docstring for likelihood"""
	def __init__(self,fitargs,fitpars,runbools,**kwargs):
		super(likelihood, self).__init__()

		self.verbose = kwargs.get('verbose',True)
		self.fitargs = fitargs

		# split up the boolean flags
		self.spec_bool = runbools[0]
		self.phot_bool = runbools[1]
		self.normspec_bool = runbools[2]
		self.imf_bool = runbools[3]
		self.photscale_bool = runbools[4]

		# # split up the boolean flags
		# self.spec_bool = runbools[0]
		# self.phot_bool = runbools[1]
		# self.normspec_bool = runbools[2]
		# self.oldnnbool = runbools[3]
		# self.imf_bool = runbools[4]
		# self.photscale_bool = runbools[5]

		# initialize the model generation class
		self.GM = GenMod()

		# initialize the ANN for spec and phot if user defined
		if self.spec_bool:
			self.GM._initspecnn(nnpath=fitargs['specANNpath'])
		if self.phot_bool:
			self.GM._initphotnn(self.fitargs['obs_phot'].keys(),
				nnpath=fitargs['photANNpath'])

		# determine the number of dims
		self.ndim = 0
		self.fitpars_i = []
		for pp in fitpars[0]:
			if fitpars[1][pp]:
				self.fitpars_i.append(pp)
				self.ndim += 1


		# # determine the number of dims
		# if self.spec_bool:
		# 	self.ndim = 7
		# if self.phot_bool:
		# 	if self.spec_bool:
		# 		self.ndim = 10
		# 	else:
		# 		self.ndim = 6
		# 	if self.photscale_bool:
		# 		self.ndim = self.ndim-1

		# if self.normspec_bool:
		# 	self.polyorder = self.fitargs['norm_polyorder']
		# 	if self.phot_bool:
		# 		self.ndim = 10+self.polyorder+1
		# 		if self.photscale_bool:
		# 			self.ndim = self.ndim-1
		# 	else:
		# 		self.ndim = 7+self.polyorder+1

	def lnlikefn(self,pars):
		# build the parameter dictionary
		parsdict = {pp:vv for pp,vv in zip(self.fitpars_i,pars)} 

		# # split upars based on the runbools
		# if (self.spec_bool and not self.phot_bool):
		# 	Teff,logg,FeH,aFe,radvel,rotvel,inst_R = pars[:7]
		# elif (self.spec_bool and self.phot_bool):
		# 	Teff,logg,FeH,aFe,radvel,rotvel,inst_R = pars[:7]
		# 	if self.photscale_bool:
		# 		logA,Av = pars[-2:]
		# 	else:
		# 		logR,Dist,Av = pars[-3:]
		# else:
		# 	if self.photscale_bool:
		# 		Teff,logg,FeH,aFe,logA,Av = pars
		# 	else:
		# 		Teff,logg,FeH,aFe,logR,Dist,Av = pars

		# # determine what paramters go into the spec function
		# if self.spec_bool:
		# 	specpars = [Teff,logg,FeH,aFe,radvel,rotvel,inst_R]
		# 	if self.normspec_bool:
		# 		if self.phot_bool:
		# 			if self.photscale_bool:
		# 				polypars = pars[7:-2]
		# 			else:
		# 				polypars = pars[7:-3]
		# 		else:
		# 			polypars = pars[7:]
		# 		for pp in polypars:
		# 			specpars.append(pp)
		# else:
		# 	specpars = None

		# # determine what paramters go into the phot function
		# if self.phot_bool:
		# 	if self.photscale_bool:
		# 		photpars = [Teff,logg,FeH,aFe,logA,Av]
		# 	else:
		# 		photpars = [Teff,logg,FeH,aFe,logR,Dist,Av]
		# else:
		# 	photpars = None


		if self.spec_bool:
				specpars = [parsdict[pp] for pp in ['Teff','log(g)','[Fe/H]','[a/Fe]','Vrad','Vrot','Inst_R']]
				if self.normspec_bool:
					specpars = specpars + [parsdict[pp] for pp in self.fitpars_i if 'pc' in pp]
			else:
				specpars = None

		if self.phot_bool:
			photpars = [parsdict[pp] for pp in ['Teff','log(g)','[Fe/H]','[a/Fe]']]
			if 'log(A)' in self.fitpars_i:
				photpars = photpars + [parsdict['log(A)']]
			else:
				photpars = photpars + [parsdict['log(R)'],parsdict['Dist']]
			photpars = photpars + [parsdict['Av']]
			# include Rv if user wants, else set to 3.1
			if 'Rv' in self.fitpars_i:
				photpars = photpars + [parsdict['Rv']]
			else:
				photpars = photpars + [None]
		else:
			photpars = None

		# calculate likelihood probability
		lnlike_i = self.lnlike(specpars=specpars,photpars=photpars)

		# embed the pars into self so that they can be pulled
		self.parsdict = parsdict

		if lnlike_i == np.nan:
			print(pars,lnlike_i)
		return lnlike_i

	def lnlike(self,specpars=None,photpars=None):

		if self.spec_bool:
			# generate model spectrum
			specmod = self.GM.genspec(specpars,outwave=self.fitargs['obs_wave_fit'],normspec_bool=self.normspec_bool)
			modwave_i,modflux_i = specmod

			# calc chi-square for spec
			specchi2 = np.sum( 
				[((m-o)**2.0)/(s**2.0) for m,o,s in zip(
					modflux_i,self.fitargs['obs_flux_fit'],self.fitargs['obs_eflux_fit'])])
		else:
			specchi2 = 0.0

		if self.phot_bool:
			# generate model SED
			if self.photscale_bool:
				sedmod  = self.GM.genphot_scaled(photpars)
			else:
				sedmod  = self.GM.genphot(photpars)

			# calculate chi-square for SED
			sedchi2 = np.sum(
				[((sedmod[kk]-self.fitargs['obs_phot'][kk][0])**2.0)/(self.fitargs['obs_phot'][kk][1]**2.0) 
				for kk in self.fitargs['obs_phot'].keys()]
				)
		else:
			sedchi2 = 0.0

		# return ln(like) = -0.5 * chi-square
		return -0.5*(specchi2+sedchi2)
