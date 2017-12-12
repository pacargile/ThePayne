import numpy as np

class prior(object):
	"""docstring for priors"""
	def __init__(self, inpriordict,runbools):
		super(prior, self).__init__()

		# find uniform priors and put them into a 
		# dictionary used for the prior transformation
		self.priordict = {}

		# put any additional priors into a dictionary so that
		# they can be applied in the lnprior_* functions
		self.additionalpriors = {}

		for kk in inpriordict.keys():
			for ii in inpriordict[kk].keys():
				if ii == 'uniform':
					self.priordict[kk] = inpriordict[kk]['uniform']
				else:
					try:
						self.additionalpriors[kk][ii] = inpriordict[kk][ii]
					except KeyError:
						self.additionalpriors[kk] = {ii:inpriordict[kk][ii]}

		# split up the boolean flags
		self.spec_bool = runbools[0]
		self.phot_bool = runbools[1]
		self.normspec_bool = runbools[2]

	def priortrans(self,upars):
		# split upars based on the runbools
		if (self.spec_bool and not self.phot_bool):
			Teff,logg,FeH,aFe,radvel,rotvel,inst_R = upars[:7]
		elif (self.spec_bool and self.phot_bool):
			Teff,logg,FeH,aFe,radvel,rotvel,inst_R = upars[:7]
			logR,Dist,Av = upars[-3:]
		else:
			Teff,logg,FeH,logR,Dist,Av = upars

		# determine what paramters go into the spec function and 
		# calculate prior transformation for spectrum
		if self.spec_bool:
			uspecpars = [Teff,logg,FeH,aFe,radvel,rotvel,inst_R]

			if self.normspec_bool:
				if self.phot_bool:
					upolypars = upars[7:-3]
				else:
					upolypars = upars[7:]
				for pp in upolypars:
					uspecpars.append(pp)
			specPT = self.priortrans_spec(uspecpars)
		else:
			specPT = []

		# determine what paramters go into the phot function and 
		# calcuate prior transformation for SED
		if self.phot_bool:
			uphotpars = [Teff,logg,FeH,logR,Dist,Av]
			photPT = self.priortrans_phot(uphotpars)
		else:
			photPT = []

		# return prior transformed parameters
		outpars = specPT + photPT
		return outpars

	def priortrans_spec(self,upars):
		# split up scaled parameters
		uTeff   = upars[0]
		ulogg   = upars[1]
		uFeH    = upars[2]
		uaFe    = upars[3]
		uradvel = upars[4]
		urotvel = upars[5]
		uinst_R = upars[6]
	
		# calcuate transformation from prior volume to parameter for all modeled parameters
		# EVENTUALLY TAKE IN ANN FILE AND DETERMINE DEFAULT GRID LIMITS FOR UNIFORM PRIORS

		if 'Teff' in self.priordict.keys():
			Teff = (max(self.priordict['Teff'])-min(self.priordict['Teff']))*uTeff + min(self.priordict['Teff'])
		else:
			# Teff    = ((10.0**self.PP.NN['x_max'][0])-(10.0**self.PP.NN['x_min'][0]))*uTeff + (10.0**self.PP.NN['x_min'][0])
			Teff = (17000.0 - 3000.0)*uTeff + 3000.0

		if 'log(g)' in self.priordict.keys():
			logg = (max(self.priordict['log(g)'])-min(self.priordict['log(g)']))*ulogg + min(self.priordict['log(g)'])
		else:
			# logg    = (self.PP.NN['x_max'][1]-self.PP.NN['x_min'][1])*ulogg + self.PP.NN['x_min'][1]
			logg = (5.5 - -1.0)*ulogg + -1.0

		if '[Fe/H]' in self.priordict.keys():
			FeH = (max(self.priordict['[Fe/H]'])-min(self.priordict['[Fe/H]']))*uFeH + min(self.priordict['[Fe/H]'])
		else:
			# FeH     = (self.PP.NN['x_max'][2]-self.PP.NN['x_min'][2])*uFeH + self.PP.NN['x_min'][2]
			FeH  = (0.5 - -2.0)*uFeH + -2.0

		if '[a/Fe]' in self.priordict.keys():
			aFe = (max(self.priordict['[a/Fe]'])-min(self.priordict['[a/Fe]']))*uaFe + min(self.priordict['[a/Fe]'])
		else:
			# aFe     = (self.PP.NN['x_max'][3]-self.PP.NN['x_min'][3])*uaFe + self.PP.NN['x_min'][3]
			aFe = (0.4 - 0.0)*uaFe + 0.0

		if 'Vrad' in self.priordict.keys():
			radvel = (max(self.priordict['Vrad'])-min(self.priordict['Vrad']))*uradvel + min(self.priordict['Vrad'])
		else:
			radvel  = (400.0 - -400.0)*uradvel + -400.0

		if 'Vrot' in self.priordict.keys():
			rotvel = (max(self.priordict['Vrot'])-min(self.priordict['Vrot']))*urotvel + min(self.priordict['Vrot'])
		else:
			rotvel  = (300.0 - 0.0)*urotvel + 0.0

		if 'Inst_R' in self.priordict.keys():
			inst_R = (max(self.priordict['Inst_R'])-min(self.priordict['Inst_R']))*uinst_R + min(self.priordict['Inst_R'])
		else:
			inst_R = (42000.0-10000.0)*uinst_R + 10000.0

		outarr = [Teff,logg,FeH,aFe,radvel,rotvel,inst_R]

		# if fitting a blaze function, do transformation for polycoef
		if self.normspec_bool:
			uspec_scale = upars[7]
			upolycoef = upars[8:]

			# spec_scale = (2.0*self.fitargs['obs_flux_fit'].max()-0.0)*uspec_scale + 0.0
			spec_scale = (0.1-0.0)*uspec_scale + 0.0
			outarr.append(spec_scale)

			# use a 5-sigma limit on uniform priors for polycoef
			for ii,upolycoef_i in enumerate(upolycoef):
				pcmax = self.polycoefarr[ii][0]+5.0*self.polycoefarr[ii][1]
				pcmin = self.polycoefarr[ii][0]-5.0*self.polycoefarr[ii][1]
				polycoef_i = (pcmax-pcmin)*upolycoef_i + pcmin
				outarr.append(polycoef_i)

		return outarr

	def priortrans_phot(self,upars):

		# if only fitting the SED, pull Teff/logg/FeH and do prior transformation
		if not self.spec_bool:
			uTeff   = upars[0]
			ulogg   = upars[1]
			uFeH    = upars[2]

			if 'Teff' in self.priordict.keys():
				Teff = (max(self.priordict['Teff'])-min(self.priordict['Teff']))*uTeff + min(self.priordict['Teff'])
			else:
				Teff = (17000.0 - 3000.0)*uTeff + 3000.0

			if 'log(g)' in self.priordict.keys():
				logg = (max(self.priordict['log(g)'])-min(self.priordict['log(g)']))*ulogg + min(self.priordict['log(g)'])
			else:
				logg = (5.5 - -1.0)*ulogg + -1.0

			if '[Fe/H]' in self.priordict.keys():
				FeH = (max(self.priordict['[Fe/H]'])-min(self.priordict['[Fe/H]']))*uFeH + min(self.priordict['[Fe/H]'])
			else:
				FeH  = (0.5 - -2.0)*uFeH + -4.0
			
			outarr = [Teff,logg,FeH]
		else:
			outarr = []

		# pull SED only parameters and do prior transformation
		ulogR  = upars[3]
		uDist = upars[4]
		uAv   = upars[5]

		if 'log(R)' in self.priordict.keys():
			logR = (max(self.priordict['log(R)'])-min(self.priordict['log(R)']))*ulogR + min(self.priordict['log(R)'])
		else:
			logR = (3.0 - -2.0)*ulogR + -2.0

		if 'Dist' in self.priordict.keys():
			Dist = (max(self.priordict['Dist'])-min(self.priordict['Dist']))*uDist + min(self.priordict['Dist'])
		else:
			Dist = (100000.0 - 0.0)*uDist + 0.0

		if 'Av' in self.priordict.keys():
			Av = (max(self.priordict['Av'])-min(self.priordict['Av']))*uAv + min(self.priordict['Av'])
		else:
			Av = (5.0-0.0)*uAv + 0.0

		outarr.append(logR)
		outarr.append(Dist)
		outarr.append(Av)

		return outarr

	def lnpriorfn(self,pars):
		# check to see if any priors in additionalprior dictionary,
		# save time by quickly returning zero if there are none
		if len(self.additionalpriors.keys()) == 0:
			return 0.0

		# split pars based on the runbools
		if (self.spec_bool and not self.phot_bool):
			Teff,logg,FeH,aFe,radvel,rotvel,inst_R = pars[:7]
		elif (self.spec_bool and self.phot_bool):
			Teff,logg,FeH,aFe,radvel,rotvel,inst_R = pars[:7]
			logR,Dist,Av = pars[-3:]
		else:
			Teff,logg,FeH,logR,Dist,Av = pars

		# determine what paramters go into the spec function
		if self.spec_bool:
			specpars = [Teff,logg,FeH,aFe,radvel,rotvel,inst_R]
			if self.normspec_bool:
				if self.phot_bool:
					polypars = pars[7:-3]
				else:
					polypars = pars[7:]
				for pp in polypars:
					specpars.append(pp)
			lnP_spec = self.lnprior_spec(specpars)
			if lnP_spec == -np.inf:
				return -np.inf
		else:
			lnP_spec = 0.0

		# determine what paramters go into the phot function
		if self.phot_bool:
			photpars = [Teff,logg,FeH,logR,Dist,Av]
			lnP_phot = self.lnprior_phot(photpars)
			if lnP_phot == -np.inf:
				return -np.inf
		else:
			lnP_phot = 0.0

		# sun prior probabilities for spec and SED
		lnprior_i = lnP_spec + lnP_phot

		return lnprior_i


	def lnprior_spec(self,pars,verbose=True):
		lnprior = 0.0

		pardict = {}
		pardict['Teff']   = pars[0]
		pardict['log(g)'] = pars[1]
		pardict['[Fe/H]'] = pars[2]
		pardict['[a/Fe]'] = pars[3]
		pardict['Vrad']   = pars[4]
		pardict['Vrot']   = pars[5]
		pardict['Inst_R'] = pars[6]

		# check to see if any of these parameter are included in additionalpriors dict
		if len(self.additionalpriors.keys()) > 0:
			for kk in self.additionalpriors.keys():
				# check to see if additional prior is for a spectroscopic parameter
				if kk in ['Teff','log(g)','[Fe/H]','[a/Fe]','Vrad','Vrot','Inst_R']:
					# if prior is Gaussian
					if 'gaussian' in self.additionalpriors[kk].keys():
						lnprior += -0.5 * (((pardict[kk]-self.additionalpriors[kk]['gaussian'][0])**2.0)/
							(self.additionalpriors[kk]['gaussian'][1]**2.0))
					elif 'flat' in self.additionalpriors[kk].keys():
						if ((pardict[kk] < self.additionalpriors[kk]['flat'][0]) or 
							(pardict[kk] > self.additionalpriors[kk]['flat'][1])):
							return -np.inf
					elif 'beta' in self.additionalpriors[kk].keys():
						raise IOError('Beta Prior not implimented yet!!!')
					elif 'log-normal' in self.additionalpriors[kk].keys():
						raise IOError('Log-Normal Prior not implimented yet!!!')
					else:
						pass

		if self.normspec_bool:
			spec_scale = pars[7]
			polycoef = pars[8:]

			# for pp in polycoef:
			# 	lnprior += -np.log(np.abs(pp))

			# for kk,pp in enumerate(polycoef):
			# 	lnprior += -0.5 * ((pp-self.polycoefarr[kk][0])**2.0) / ((0.1*self.polycoefarr[kk][0])**2.0)


		return lnprior

	def lnprior_phot(self,pars,verbose=True):
		lnprior = 0.0

		# pull out the pars and put into a dictionary
		pardict = {}
		pardict['Teff']   = pars[0]
		pardict['log(g)'] = pars[1]
		pardict['[Fe/H]'] = pars[2]
		pardict['log(R)'] = pars[3]
		pardict['Dist']   = pars[4]
		pardict['Av']     = pars[5]

		# check to see if any of these parameter are included in additionalpriors dict
		if len(self.additionalpriors.keys()) > 0:
			for kk in self.additionalpriors.keys():
				# check to see if additional prior is for a spectroscopic parameter
				if kk in ['Teff','log(g)','[Fe/H]','log(R)','Dist','Av']:
					# if prior is Gaussian
					if 'gaussian' in self.additionalpriors[kk].keys():
						lnprior += -0.5 * (((pardict[kk]-self.additionalpriors[kk]['gaussian'][0])**2.0)/
							(self.additionalpriors[kk]['gaussian'][1]**2.0))
					elif 'flat' in self.additionalpriors[kk].keys():
						if ((pardict[kk] < self.additionalpriors[kk]['flat'][0]) or 
							(pardict[kk] > self.additionalpriors[kk]['flat'][1])):
							return -np.inf
					elif 'beta' in self.additionalpriors[kk].keys():
						raise IOError('Beta Prior not implimented yet!!!')
					elif 'log-normal' in self.additionalpriors[kk].keys():
						raise IOError('Log-Normal Prior not implimented yet!!!')
					else:
						pass

		return lnprior