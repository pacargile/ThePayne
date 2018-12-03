import numpy as np

class prior(object):
	"""docstring for priors"""
	def __init__(self, inpriordict,fitpars,runbools):
		super(prior, self).__init__()

		# determine the number of dims
		self.ndim = 0
		self.fitpars_i = []
		for pp in fitpars[0]:
			if fitpars[1][pp]:
				self.fitpars_i.append(pp)
				self.ndim += 1

		# find uniform priors and put them into a 
		# dictionary used for the prior transformation
		self.priordict = {}

		# put any additional priors into a dictionary so that
		# they can be applied in the lnprior_* functions
		self.additionalpriors = {}

		for kk in inpriordict.keys():
			if kk == 'blaze_coeff':
				self.polycoefarr = inpriordict['blaze_coeff']
			elif kk == 'IMF':
				self.imf = inpriordict['IMF']['IMF_type']
			else:
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
		self.imf_bool = runbools[3]
		self.photscale_bool = runbools[4]

	def priortrans(self,upars):
		# build the parameter dictionary
		uparsdict = {pp:vv for pp,vv in zip(self.fitpars_i,upars)} 

		if self.spec_bool:
			specPT = self.priortrans_spec(uparsdict)
		else:
			specPT = {}

		if self.phot_bool:
			photPT = self.priortrans_phot(uparsdict)
		else:
			photPT = {}

		outputPT = {**specPT,**photPT}

		return [outputPT[pp] for pp in self.fitpars_i]


	def priortrans_spec(self,upars):
		outdict = {}

		if 'Teff' in upars.keys():
			uTeff = upars['Teff']
			if 'Teff' in self.priordict.keys():
				Teff = (max(self.priordict['Teff'])-min(self.priordict['Teff']))*uTeff + min(self.priordict['Teff'])
			else:
				# Teff    = ((10.0**self.PP.NN['x_max'][0])-(10.0**self.PP.NN['x_min'][0]))*uTeff + (10.0**self.PP.NN['x_min'][0])
				Teff = (17000.0 - 3000.0)*uTeff + 3000.0
			outdict['Teff'] = Teff

		if 'log(g)' in upars.keys():
			ulogg = upars['log(g)']
			if 'log(g)' in self.priordict.keys():
				logg = (max(self.priordict['log(g)'])-min(self.priordict['log(g)']))*ulogg + min(self.priordict['log(g)'])
			else:
				# logg    = (self.PP.NN['x_max'][1]-self.PP.NN['x_min'][1])*ulogg + self.PP.NN['x_min'][1]
				logg = (5.5 - -1.0)*ulogg + -1.0
			outdict['log(g)'] = logg

		if '[Fe/H]' in upars.keys():
			uFeH = upars['[Fe/H]']
			if '[Fe/H]' in self.priordict.keys():
				FeH = (max(self.priordict['[Fe/H]'])-min(self.priordict['[Fe/H]']))*uFeH + min(self.priordict['[Fe/H]'])
			else:
				# FeH     = (self.PP.NN['x_max'][2]-self.PP.NN['x_min'][2])*uFeH + self.PP.NN['x_min'][2]
				FeH  = (0.5 - -2.0)*uFeH + -2.0
			outdict['[Fe/H]'] = FeH

		if '[a/Fe]' in upars.keys():
			uaFe = upars['[a/Fe]']
			if '[a/Fe]' in self.priordict.keys():
				aFe = (max(self.priordict['[a/Fe]'])-min(self.priordict['[a/Fe]']))*uaFe + min(self.priordict['[a/Fe]'])
			else:
				# aFe     = (self.PP.NN['x_max'][3]-self.PP.NN['x_min'][3])*uaFe + self.PP.NN['x_min'][3]
				aFe = (0.6 - -0.2)*uaFe + -0.2
			outdict['[a/Fe]'] = aFe

		if 'Vrad' in upars.keys():
			uradvel = upars['Vrad']
			if 'Vrad' in self.priordict.keys():
				radvel = (max(self.priordict['Vrad'])-min(self.priordict['Vrad']))*uradvel + min(self.priordict['Vrad'])
			else:
				radvel  = (400.0 - -400.0)*uradvel + -400.0
			outdict['Vrad'] = radvel

		if 'Vrot' in upars.keys():
			urotvel = upars['Vrot']
			if 'Vrot' in self.priordict.keys():
				rotvel = (max(self.priordict['Vrot'])-min(self.priordict['Vrot']))*urotvel + min(self.priordict['Vrot'])
			else:
				rotvel  = (300.0 - 0.0)*urotvel + 0.0
			outdict['Vrot'] = rotvel

		if 'Inst_R' in upars.keys():
			uinst_R = upars['Inst_R']
			if 'Inst_R' in self.priordict.keys():
				inst_R = (max(self.priordict['Inst_R'])-min(self.priordict['Inst_R']))*uinst_R + min(self.priordict['Inst_R'])
			else:
				inst_R = (60000.0-10000.0)*uinst_R + 10000.0
			outdict['Inst_R'] = inst_R

		# if fitting a blaze function, do transformation for polycoef
		pcarr = [x_i for x_i in upars.keys() if 'pc' in x_i]
		if len(pcarr) > 0:
			for pc_i in pcarr:
				if pc_i == 'pc_0':
					uspec_scale = upars['pc_0']
					outdict['pc_0'] = (1.0-0.0)*uspec_scale + 0.0
				else:
					pcind = int(pc_i.split('_')[-1])
					pcmax = self.polycoefarr[pcind][0]+5.0*self.polycoefarr[pcind][1]
					pcmin = self.polycoefarr[pcind][0]-5.0*self.polycoefarr[pcind][1]
					outdict[pc_i] = (pcmax-pcmin)*upars[pc_i] + pcmin

		return outdict


	def priortrans_phot(self,upars):

		outdict = {}

		# if only fitting the SED, pull Teff/logg/FeH and do prior transformation
		if not self.spec_bool:

			if 'Teff' in upars.keys():
				uTeff = upars['Teff']
				if 'Teff' in self.priordict.keys():
					Teff = (max(self.priordict['Teff'])-min(self.priordict['Teff']))*uTeff + min(self.priordict['Teff'])
				else:
					# Teff    = ((10.0**self.PP.NN['x_max'][0])-(10.0**self.PP.NN['x_min'][0]))*uTeff + (10.0**self.PP.NN['x_min'][0])
					Teff = (17000.0 - 3000.0)*uTeff + 3000.0
				outdict['Teff'] = Teff

			if 'log(g)' in upars.keys():
				ulogg = upars['log(g)']
				if 'log(g)' in self.priordict.keys():
					logg = (max(self.priordict['log(g)'])-min(self.priordict['log(g)']))*ulogg + min(self.priordict['log(g)'])
				else:
					# logg    = (self.PP.NN['x_max'][1]-self.PP.NN['x_min'][1])*ulogg + self.PP.NN['x_min'][1]
					logg = (5.5 - -1.0)*ulogg + -1.0
				outdict['log(g)'] = logg

			if '[Fe/H]' in upars.keys():
				uFeH = upars['[Fe/H]']
				if '[Fe/H]' in self.priordict.keys():
					FeH = (max(self.priordict['[Fe/H]'])-min(self.priordict['[Fe/H]']))*uFeH + min(self.priordict['[Fe/H]'])
				else:
					# FeH     = (self.PP.NN['x_max'][2]-self.PP.NN['x_min'][2])*uFeH + self.PP.NN['x_min'][2]
					FeH  = (0.5 - -2.0)*uFeH + -2.0
				outdict['[Fe/H]'] = FeH

			if '[a/Fe]' in upars.keys():
				uaFe = upars['[a/Fe]']
				if '[a/Fe]' in self.priordict.keys():
					aFe = (max(self.priordict['[a/Fe]'])-min(self.priordict['[a/Fe]']))*uaFe + min(self.priordict['[a/Fe]'])
				else:
					# aFe     = (self.PP.NN['x_max'][3]-self.PP.NN['x_min'][3])*uaFe + self.PP.NN['x_min'][3]
					aFe = (0.6 - -0.2)*uaFe + -0.2
				outdict['[a/Fe]'] = aFe


		if 'log(A)' in upars.keys():
			ulogA = upars['log(A)']
			if 'log(A)' in self.priordict.keys():
				logA = (max(self.priordict['log(A)'])-min(self.priordict['log(A)']))*ulogA + min(self.priordict['log(A)'])
			else:
				logA = (7.0 - -3.0)*ulogA + -3.0
			outdict['log(A)'] = logA

		if 'log(R)' in upars.keys():
			ulogR = upars['log(R)']
			if 'log(R)' in self.priordict.keys():
				logR = (max(self.priordict['log(R)'])-min(self.priordict['log(R)']))*ulogR + min(self.priordict['log(R)'])
			else:
				logR = (3.0 - -2.0)*ulogR + -2.0
			outdict['log(R)'] = logR

		if 'Dist' in upars.keys():
			uDist = upars['Dist']
			if 'Dist' in self.priordict.keys():
				Dist = (max(self.priordict['Dist'])-min(self.priordict['Dist']))*uDist + min(self.priordict['Dist'])
			else:
				Dist = (100000.0 - 0.0)*uDist + 0.0
			outdict['Dist'] = Dist

		if 'Av' in upars.keys():
			uAv = upars['Av']
			if 'Av' in self.priordict.keys():
				Av = (max(self.priordict['Av'])-min(self.priordict['Av']))*uAv + min(self.priordict['Av'])
			else:
				Av = (5.0-0.0)*uAv + 0.0
			outdict['Av'] = Av

		if 'Rv' in upars.keys():
			uRv = upars['Rv']
			if 'Rv' in self.priordict.keys():
				Rv = (max(self.priordict['Rv'])-min(self.priordict['Rv']))*uRv + min(self.priordict['Rv'])
			else:
				Rv = (5.0-2.0)*uRv + 2.0
			outdict['Rv'] = Rv

		return outdict



	def lnpriorfn(self,pars):
		# check to see if any priors in additionalprior dictionary,
		# save time by quickly returning zero if there are none
		if len(self.additionalpriors.keys()) == 0:
			return 0.0

		# determine if user passed a dictionary or a list
		if isinstance(pars,list):
			# build the parameter dictionary
			parsdict = {pp:vv for pp,vv in zip(self.fitpars_i,pars)} 
		else:
			parsdict = pars

		if self.spec_bool:
			specPrior = self.lnprior_spec(parsdict)
		else:
			specPrior = 0.0

		if self.phot_bool:
			photPrior = self.lnprior_phot(parsdict)
		else:
			photPrior = 0.0

		return specPrior + photPrior
	def lnprior_spec(self,pardict,verbose=True):
		lnprior = 0.0

		# check to see if any of the parameter are included in additionalpriors dict
		if len(self.additionalpriors.keys()) > 0:
			for kk in self.additionalpriors.keys():
				# check to see if additional prior is for a spectroscopic parameter
				if kk in ['Teff','log(g)','[Fe/H]','[a/Fe]','Vrad','Vrot','Inst_R']:
					# if prior is Gaussian
					if 'gaussian' in self.additionalpriors[kk].keys():
						lnprior += -0.5 * (((pardict[kk]-self.additionalpriors[kk]['gaussian'][0])**2.0)/
							(self.additionalpriors[kk]['gaussian'][1]**2.0))
					elif 'uniform' in self.additionalpriors[kk].keys():
						if ((pardict[kk] < self.additionalpriors[kk]['uniform'][0]) or 
							(pardict[kk] > self.additionalpriors[kk]['uniform'][1])):
							return -np.inf
					elif 'beta' in self.additionalpriors[kk].keys():
						raise IOError('Beta Prior not implimented yet!!!')
					elif 'log-normal' in self.additionalpriors[kk].keys():
						raise IOError('Log-Normal Prior not implimented yet!!!')
					else:
						pass

		# if fitting a blaze function, then check for additional priors
		if self.normspec_bool:
			for pp in self.fitpars_i:
				if pp[:2] == 'pc_':
					lnprior += -0.5 * ((parsdict[pp]/self.polycoefarr[kk][1])**2.0)

		return lnprior

	def lnprior_phot(self,pardict,verbose=True):
		lnprior = 0.0

		pardict_i = {}
		# if only fitting the SED, pull Teff/logg/FeH and do prior 
		if not self.spec_bool:
			pardict_i['Teff']   = pardict['Teff']
			pardict_i['log(g)'] = pardict['log(g)']
			pardict_i['[Fe/H]'] = pardict['[Fe/H]']
			pardict_i['[a/Fe]'] = pardict['[a/Fe]']
		if 'log(R)' in self.fitpars_i:
			pardict_i['log(R)'] = pardict['log(R)']
		if 'Dist' in self.fitpars_i:
			pardict_i['Dist'] = pardict['Dist']
		if 'log(A)' in self.fitpars_i:
			pardict_i['log(A)'] = pardict['log(A)']
		if 'Av' in self.fitpars_i:
			pardict_i['Av'] = pardict['Av']
		if 'Dist' in self.fitpars_i:
			pardict_i['Parallax'] = 1000.0/pardict['Dist']

		# check to see if any of these parameter are included in additionalpriors dict
		if len(self.additionalpriors.keys()) > 0:
			for kk in self.additionalpriors.keys():
				if kk in ['Teff','log(g)','[Fe/H]','[a/Fe]','log(R)','Dist','log(A)','Av','Parallax']:
					# if prior is Gaussian
					if 'gaussian' in self.additionalpriors[kk].keys():
						lnprior += -0.5 * (((pardict_i[kk]-self.additionalpriors[kk]['gaussian'][0])**2.0)/
							(self.additionalpriors[kk]['gaussian'][1]**2.0))
					elif 'uniform' in self.additionalpriors[kk].keys():
						if ((pardict_i[kk] < self.additionalpriors[kk]['uniform'][0]) or 
							(pardict_i[kk] > self.additionalpriors[kk]['uniform'][1])):
							return -np.inf
					elif 'beta' in self.additionalpriors[kk].keys():
						raise IOError('Beta Prior not implimented yet!!!')
					elif 'log-normal' in self.additionalpriors[kk].keys():
						raise IOError('Log-Normal Prior not implimented yet!!!')
					else:
						pass

		# apply IMF pior
		if self.imf_bool:
			logM = pardict_i['log(g)'] + 2.0 * pardict_i['log(R)']
			if self.imf == 'Kroupa':
				P_mass = self.kroupa(logM)
			else:
				raise ValueError('Only Kroupa IMF available currently')
			lnprior += np.log(P_mass)

		return lnprior

	def kroupa(self,logM):
		"""
		calculate P(M|Kroupa IMF)
		"""
		m = 10.0**logM
		if m < 0.5:
			alpha = 1.3
		else:
			alpha = 2.3
		gamma = -1.0*alpha
		A = (1+gamma)/((100.0**(1.0+gamma))-(0.1**(1.0+gamma)))
		return A*(m**gamma)

