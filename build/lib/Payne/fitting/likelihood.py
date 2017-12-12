import numpy as np
from .genmod import GenMod

class likelihood(object):
	"""docstring for likelihood"""
	def __init__(self,fitargs,runbools,**kwargs):
		super(likelihood, self).__init__()

		self.verbose = kwargs.get('verbose',True)

		self.GM = GenMod()

		self.fitargs = fitargs
		self.spec_bool,self.phot_bool,self.normspec_bool = runbools

		if self.spec_bool:
			self.GM._initspecnn(nnpath=fitargs['specANNpath'])

		if self.phot_bool:
			self.GM._initphotnn(self.fitargs['obs_phot'].keys(),
				nnpath=fitargs['photANNpath'])

		# determine the number of dims
		if self.spec_bool:
			self.ndim = 7

		if self.phot_bool:
			if self.spec_bool:
				self.ndim = 10
			else:
				self.ndim = 6

		if self.normspec_bool:
			self.polyorder = self.fitargs['norm_polyorder']
			if self.phot_bool:
				self.ndim = 10+self.polyorder+1
			else:
				self.ndim = 7+self.polyorder+1

	def lnprob(self,pars):
		# split upars based on the runbools
		if (self.spec_bool and not self.phot_bool):
			Teff,logg,FeH,aFe,radvel,rotvel,inst_R = pars[:7]
		elif (self.spec_bool and self.phot_bool):
			Teff,logg,FeH,aFe,radvel,rotvel,inst_R = pars[:7]
			logR,Dist,Av = pars[-3:]
		else:
			Teff,logg,FeH,logR,Dist,Av = pars

		# if (self.ndim == 7) or (self.ndim == 7+self.polyorder+1):
		# 	Teff,logg,FeH,aFe,radvel,rotvel,inst_R = pars[:7]
		# elif (self.ndim == 10) or (self.ndim == 10+self.polyorder+1):
		# 	Teff,logg,FeH,aFe,radvel,rotvel,inst_R = pars[:7]
		# 	logR,Dist,Av = pars[-3:]
		# else:
		# 	#self.ndim == 6
		# 	Teff,logg,FeH,logR,Dist,Av = pars

		if self.spec_bool:
			specpars = [Teff,logg,FeH,aFe,radvel,rotvel,inst_R]
			if self.normspec_bool:
				if self.phot_bool:
					polypars = pars[7:-3]
				else:
					polypars = pars[7:]
				for pp in polypars:
					specpars.append(pp)
		else:
			specpars = None

		if self.phot_bool:
			photpars = [Teff,logg,FeH,logR,Dist,Av]
		else:
			photpars = None

		# if self.spec_bool:
		# 	lnprior_spec_i = self.lnprior_spec(specpars)
		# else:
		# 	lnprior_spec_i = 0.0

		# if self.phot_bool:
		# 	lnprior_phot_i = self.lnprior_phot(photpars)
		# else:
		# 	lnprior_phot_i = 0.0


		# if (lnprior_spec_i == -np.inf) or (lnprior_phot_i == -np.inf):
		# 	return -np.inf

		lnlike_i = self.lnlike(specpars=specpars,photpars=photpars)

		# lnprob_i = lnlike_i + lnprior_spec_i + lnprior_phot_i
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
			sedmod  = self.GM.genphot(photpars)

			sedchi2 = np.sum(
				[((sedmod[kk]-self.fitargs['obs_phot'][kk][0])**2.0)/(self.fitargs['obs_phot'][kk][1]**2.0) 
				for kk in self.fitargs['obs_phot'].keys()]
				)
		else:
			sedchi2 = 0.0

		return -0.5*(specchi2+sedchi2)
