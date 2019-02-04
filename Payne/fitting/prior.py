import numpy as np
from scipy.stats import norm, truncnorm
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
		self.priordict['uniform'] = {}
		self.priordict['gaussian'] = {}
		self.priordict['tgaussian'] = {}

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
						self.priordict['uniform'][kk] = inpriordict[kk]['uniform']
					elif ii == 'pv_gaussian':
						self.priordict['gaussian'][kk] = inpriordict[kk]['pv_gaussian']
					elif ii == 'pv_tgaussian':
						self.priordict['tgaussian'][kk] = inpriordict[kk]['pv_tgaussian']
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

		# dictionary of default parameter ranges
		self.defaultpars = {}
		self.defaultpars['Teff']   = [3000.0,17000.0]
		self.defaultpars['log(g)'] = [-1.0,5.5]
		self.defaultpars['[Fe/H]'] = [-4.0,0.5]
		self.defaultpars['[a/Fe]'] = [-0.2,0.6]
		self.defaultpars['Vrad']   = [-700.0,700.0]
		self.defaultpars['Vrot']   = [0,300.0]
		self.defaultpars['Inst_R'] = [10000.0,60000.0]
		self.defaultpars['log(A)'] = [-3.0,7.0]
		self.defaultpars['log(R)'] = [-2.0,3.0]
		self.defaultpars['Dist']   = [0.0,100000.0]
		self.defaultpars['Av']     = [0.0,5.0]
		self.defaultpars['Rv']     = [2.0,5.0]


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

		for namepar in ['Teff','log(g)','[Fe/H]','[a/Fe]','Vrad','Vrot','Inst_R']:
			if namepar in upars.keys():
				upars_i = upars[namepar]
				if namepar in self.priordict['uniform'].keys():
					par_i = (
						(max(self.priordict['uniform'][namepar])-min(self.priordict['uniform'][namepar]))*upars_i + 
						min(self.priordict['uniform'][namepar])
						)
				elif namepar in self.priordict['gaussian'].keys():
					par_i = norm.ppf(upars_i,loc=self.priordict['gaussian'][namepar][0],scale=self.priordict['gaussian'][namepar][1])

				elif namepar in self.priordict['tgaussian'].keys():
					a = (self.priordict['tgaussian'][namepar][0] - self.priordict['tgaussian'][namepar][2]) / self.priordict['tgaussian'][namepar][3]
					b = (self.priordict['tgaussian'][namepar][1] - self.priordict['tgaussian'][namepar][2]) / self.priordict['tgaussian'][namepar][3]
					par_i = truncnorm.ppf(upars_i,a,b,loc=self.priordict['tgaussian'][namepar][2],scale=self.priordict['tgaussian'][namepar][3])
				else:
					par_i = (self.defaultpars[namepar][1]-self.defaultpars[namepar][0])*upars_i + self.defaultpars[namepar][0]

				outdict[namepar] = par_i

		# if 'Teff' in upars.keys():
		# 	uTeff = upars['Teff']
		# 	if 'Teff' in self.priordict['uniform'].keys():
		# 		Teff = (
		# 			(max(self.priordict['uniform']['Teff'])-min(self.priordict['uniform']['Teff']))*uTeff + 
		# 			min(self.priordict['uniform']['Teff']))
		# 	elif 'Teff' in self.priordict['gaussian'].keys():
		# 		Teff = norm.ppf(uTeff,loc=self.priordict['gaussian']['Teff'][0],scale=self.priordict['gaussian']['Teff'][1])
		# 	else:
		# 		# Teff    = ((10.0**self.PP.NN['x_max'][0])-(10.0**self.PP.NN['x_min'][0]))*uTeff + (10.0**self.PP.NN['x_min'][0])
		# 		Teff = (17000.0 - 3000.0)*uTeff + 3000.0

		# 	outdict['Teff'] = Teff

		# if 'log(g)' in upars.keys():
		# 	ulogg = upars['log(g)']
		# 	if 'log(g)' in self.priordict['uniform'].keys():
		# 		logg = (
		# 			(max(self.priordict['uniform']['log(g)'])-min(self.priordict['uniform']['log(g)']))*ulogg + 
		# 			min(self.priordict['uniform']['log(g)']))
		# 	elif 'log(g)' in self.priordict['gaussian'].keys():
		# 		logg = norm.ppf(uTeff,loc=self.priordict['gaussian']['log(g)'][0],scale=self.priordict['gaussian']['log(g)'][1])
		# 	else:
		# 		# logg    = (self.PP.NN['x_max'][1]-self.PP.NN['x_min'][1])*ulogg + self.PP.NN['x_min'][1]
		# 		logg = (5.5 - -1.0)*ulogg + -1.0
		# 	outdict['log(g)'] = logg

		# if '[Fe/H]' in upars.keys():
		# 	uFeH = upars['[Fe/H]']
		# 	if '[Fe/H]' in self.priordict['uniform'].keys():
		# 		FeH = (
		# 			(max(self.priordict['uniform']['[Fe/H]'])-min(self.priordict['uniform']['[Fe/H]']))*uFeH + 
		# 			min(self.priordict['uniform']['[Fe/H]']))
		# 	elif '[Fe/H]' in self.priordict['gaussian'].keys():
		# 		FeH = norm.ppf(uFeH,loc=self.priordict['gaussian']['[Fe/H]'][0],scale=self.priordict['gaussian']['[Fe/H]'][1])
		# 	else:
		# 		# FeH     = (self.PP.NN['x_max'][2]-self.PP.NN['x_min'][2])*uFeH + self.PP.NN['x_min'][2]
		# 		FeH  = (0.5 - -2.0)*uFeH + -2.0
		# 	outdict['[Fe/H]'] = FeH

		# if '[a/Fe]' in upars.keys():
		# 	uaFe = upars['[a/Fe]']
		# 	if '[a/Fe]' in self.priordict['uniform'].keys():
		# 		aFe = (
		# 			(max(self.priordict['uniform']['[a/Fe]'])-min(self.priordict['uniform']['[a/Fe]']))*uaFe + 
		# 			min(self.priordict['uniform']['[a/Fe]']))
		# 	elif '[a/Fe]' in self.priordict['gaussian'].keys():
		# 		aFe = norm.ppf(uaFe,loc=self.priordict['gaussian']['[a/Fe]'][0],scale=self.priordict['gaussian']['[a/Fe]'][1])
		# 	else:
		# 		# aFe     = (self.PP.NN['x_max'][3]-self.PP.NN['x_min'][3])*uaFe + self.PP.NN['x_min'][3]
		# 		aFe = (0.6 - -0.2)*uaFe + -0.2
		# 	outdict['[a/Fe]'] = aFe

		# if 'Vrad' in upars.keys():
		# 	uradvel = upars['Vrad']
		# 	if 'Vrad' in self.priordict['uniform'].keys():
		# 		radvel = (
		# 			(max(self.priordict['uniform']['Vrad'])-min(self.priordict['uniform']['Vrad']))*uradvel + 
		# 			min(self.priordict['uniform']['Vrad']))
		# 	elif 'Vrad' in self.priordict['gaussian'].keys():
		# 		radvel = norm.ppf(uradvel,loc=self.priordict['gaussian']['Vrad'][0],scale=self.priordict['gaussian']['Vrad'][1])
		# 	else:
		# 		radvel  = (700.0 - -700.0)*uradvel + -700.0
		# 	outdict['Vrad'] = radvel

		# if 'Vrot' in upars.keys():
		# 	urotvel = upars['Vrot']
		# 	if 'Vrot' in self.priordict['uniform'].keys():
		# 		rotvel = (
		# 			(max(self.priordict['uniform']['Vrot'])-min(self.priordict['uniform']['Vrot']))*urotvel + 
		# 			min(self.priordict['uniform']['Vrot']))
		# 	elif 'Vrot' in self.priordict['gaussian'].keys():
		# 		rotvel = norm.ppf(urotvel,loc=self.priordict['gaussian']['Vrot'][0],scale=self.priordict['gaussian']['Vrot'][1])
		# 	else:
		# 		rotvel  = (300.0 - 0.0)*urotvel + 0.0
		# 	outdict['Vrot'] = rotvel

		# if 'Inst_R' in upars.keys():
		# 	uinst_R = upars['Inst_R']
		# 	if 'Inst_R' in self.priordict['uniform'].keys():
		# 		inst_R = (
		# 			(max(self.priordict['uniform']['Inst_R'])-min(self.priordict['uniform']['Inst_R']))*uinst_R + 
		# 			min(self.priordict['uniform']['Inst_R']))
		# 	elif 'Inst_R' in self.priordict['gaussian'].keys():
		# 		inst_R = norm.ppf(uinst_R,loc=self.priordict['gaussian']['Inst_R'][0],scale=self.priordict['gaussian']['Inst_R'][1])
		# 	else:
		# 		inst_R = (60000.0-10000.0)*uinst_R + 10000.0
		# 	outdict['Inst_R'] = inst_R

		# if fitting a blaze function, do transformation for polycoef
		pcarr = [x_i for x_i in upars.keys() if 'pc' in x_i]
		if len(pcarr) > 0:
			for pc_i in pcarr:
				if pc_i == 'pc_0':
					uspec_scale = upars['pc_0']
					outdict['pc_0'] = (1.0 - -1.0)*uspec_scale - 1.0
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
			for namepar in ['Teff','log(g)','[Fe/H]','[a/Fe]']:
				if namepar in upars.keys():
					upars_i = upars[namepar]
					if namepar in self.priordict['uniform'].keys():
						par_i = (
							(max(self.priordict['uniform'][namepar])-min(self.priordict['uniform'][namepar]))*upars_i + 
							min(self.priordict['uniform'][namepar])
							)
					elif namepar in self.priordict['gaussian'].keys():
						par_i = norm.ppf(upars_i,loc=self.priordict['gaussian'][namepar][0],scale=self.priordict['gaussian'][namepar][1])

					elif namepar in self.priordict['tgaussian'].keys():
						a = (self.priordict['tgaussian'][namepar][0] - self.priordict['tgaussian'][namepar][2]) / self.priordict['tgaussian'][namepar][3]
						b = (self.priordict['tgaussian'][namepar][1] - self.priordict['tgaussian'][namepar][2]) / self.priordict['tgaussian'][namepar][3]
						par_i = truncnorm.ppf(upars_i,a,b,loc=self.priordict['tgaussian'][namepar][2],scale=self.priordict['tgaussian'][namepar][3])
					else:
						par_i = (self.defaultpars[namepar][1]-self.defaultpars[namepar][0])*upars_i + self.defaultpars[namepar][0]

					outdict[namepar] = par_i


			# if 'Teff' in upars.keys():
			# 	uTeff = upars['Teff']
			# 	if 'Teff' in self.priordict['uniform'].keys():
			# 		Teff = (
			# 			(max(self.priordict['uniform']['Teff'])-min(self.priordict['uniform']['Teff']))*uTeff + 
			# 			min(self.priordict['uniform']['Teff']))
			# 	elif 'Teff' in self.priordict['gaussian'].keys():
			# 		Teff = norm.ppf(uTeff,loc=self.priordict['gaussian']['Teff'][0],scale=self.priordict['gaussian']['Teff'][1])
			# 	else:
			# 		# Teff    = ((10.0**self.PP.NN['x_max'][0])-(10.0**self.PP.NN['x_min'][0]))*uTeff + (10.0**self.PP.NN['x_min'][0])
			# 		Teff = (17000.0 - 3000.0)*uTeff + 3000.0

			# 	outdict['Teff'] = Teff

			# if 'log(g)' in upars.keys():
			# 	ulogg = upars['log(g)']
			# 	if 'log(g)' in self.priordict['uniform'].keys():
			# 		logg = (
			# 			(max(self.priordict['uniform']['log(g)'])-min(self.priordict['uniform']['log(g)']))*ulogg + 
			# 			min(self.priordict['uniform']['log(g)']))
			# 	elif 'log(g)' in self.priordict['gaussian'].keys():
			# 		logg = norm.ppf(uTeff,loc=self.priordict['gaussian']['log(g)'][0],scale=self.priordict['gaussian']['log(g)'][1])
			# 	else:
			# 		# logg    = (self.PP.NN['x_max'][1]-self.PP.NN['x_min'][1])*ulogg + self.PP.NN['x_min'][1]
			# 		logg = (5.5 - -1.0)*ulogg + -1.0
			# 	outdict['log(g)'] = logg

			# if '[Fe/H]' in upars.keys():
			# 	uFeH = upars['[Fe/H]']
			# 	if '[Fe/H]' in self.priordict['uniform'].keys():
			# 		FeH = (
			# 			(max(self.priordict['uniform']['[Fe/H]'])-min(self.priordict['uniform']['[Fe/H]']))*uFeH + 
			# 			min(self.priordict['uniform']['[Fe/H]']))
			# 	elif '[Fe/H]' in self.priordict['gaussian'].keys():
			# 		FeH = norm.ppf(uFeH,loc=self.priordict['gaussian']['[Fe/H]'][0],scale=self.priordict['gaussian']['[Fe/H]'][1])
			# 	else:
			# 		# FeH     = (self.PP.NN['x_max'][2]-self.PP.NN['x_min'][2])*uFeH + self.PP.NN['x_min'][2]
			# 		FeH  = (0.5 - -2.0)*uFeH + -2.0
			# 	outdict['[Fe/H]'] = FeH

			# if '[a/Fe]' in upars.keys():
			# 	uaFe = upars['[a/Fe]']
			# 	if '[a/Fe]' in self.priordict['uniform'].keys():
			# 		aFe = (
			# 			(max(self.priordict['uniform']['[a/Fe]'])-min(self.priordict['uniform']['[a/Fe]']))*uaFe + 
			# 			min(self.priordict['uniform']['[a/Fe]']))
			# 	elif '[a/Fe]' in self.priordict['gaussian'].keys():
			# 		aFe = norm.ppf(uaFe,loc=self.priordict['gaussian']['[a/Fe]'][0],scale=self.priordict['gaussian']['[a/Fe]'][1])
			# 	else:
			# 		# aFe     = (self.PP.NN['x_max'][3]-self.PP.NN['x_min'][3])*uaFe + self.PP.NN['x_min'][3]
			# 		aFe = (0.6 - -0.2)*uaFe + -0.2
			# 	outdict['[a/Fe]'] = aFe
		for namepar in ['log(A)','log(R)','Dist','Av','Rv']:
			if namepar in upars.keys():
				upars_i = upars[namepar]
				if namepar in self.priordict['uniform'].keys():
					par_i = (
						(max(self.priordict['uniform'][namepar])-min(self.priordict['uniform'][namepar]))*upars_i + 
						min(self.priordict['uniform'][namepar])
						)
				elif namepar in self.priordict['gaussian'].keys():
					par_i = norm.ppf(upars_i,loc=self.priordict['gaussian'][namepar][0],scale=self.priordict['gaussian'][namepar][1])

				elif namepar in self.priordict['tgaussian'].keys():
					a = (self.priordict['tgaussian'][namepar][0] - self.priordict['tgaussian'][namepar][2]) / self.priordict['tgaussian'][namepar][3]
					b = (self.priordict['tgaussian'][namepar][1] - self.priordict['tgaussian'][namepar][2]) / self.priordict['tgaussian'][namepar][3]
					par_i = truncnorm.ppf(upars_i,a,b,loc=self.priordict['tgaussian'][namepar][2],scale=self.priordict['tgaussian'][namepar][3])
				else:
					par_i = (self.defaultpars[namepar][1]-self.defaultpars[namepar][0])*upars_i + self.defaultpars[namepar][0]

				outdict[namepar] = par_i

		# if 'log(A)' in upars.keys():
		# 	ulogA = upars['log(A)']
		# 	if 'log(A)' in self.priordict['uniform'].keys():
		# 		logA = (
		# 			(max(self.priordict['uniform']['log(A)'])-min(self.priordict['uniform']['log(A)']))*ulogA + 
		# 			min(self.priordict['uniform']['log(A)']))
		# 	elif 'log(A)' in self.priordict['gaussian'].keys():
		# 		logA = norm.ppf(ulogA,loc=self.priordict['gaussian']['log(A)'][0],scale=self.priordict['gaussian']['log(A)'][1])
		# 	else:
		# 		logA = (7.0 - -3.0)*ulogA + -3.0
		# 	outdict['log(A)'] = logA

		# if 'log(R)' in upars.keys():
		# 	ulogR = upars['log(R)']
		# 	if 'log(R)' in self.priordict['uniform'].keys():
		# 		logR = (
		# 			(max(self.priordict['uniform']['log(R)'])-min(self.priordict['uniform']['log(R)']))*ulogR + 
		# 			min(self.priordict['uniform']['log(R)']))
		# 	elif 'log(R)' in self.priordict['gaussian'].keys():
		# 		logR = norm.ppf(ulogR,loc=self.priordict['gaussian']['log(R)'][0],scale=self.priordict['gaussian']['log(R)'][1])
		# 	else:
		# 		logR = (3.0 - -2.0)*ulogR + -2.0
		# 	outdict['log(R)'] = logR

		# if 'Dist' in upars.keys():
		# 	uDist = upars['Dist']
		# 	if 'Dist' in self.priordict['uniform'].keys():
		# 		Dist = (
		# 			(max(self.priordict['uniform']['Dist'])-min(self.priordict['uniform']['Dist']))*uDist + 
		# 			min(self.priordict['uniform']['Dist']))
		# 	elif 'Dist' in self.priordict['gaussian'].keys():
		# 		Dist = norm.ppf(uDist,loc=self.priordict['gaussian']['Dist'][0],scale=self.priordict['gaussian']['Dist'][1])
		# 	else:
		# 		Dist = (100000.0 - 0.0)*uDist + 0.0
		# 	outdict['Dist'] = Dist

		# if 'Av' in upars.keys():
		# 	uAv = upars['Av']
		# 	if 'Av' in self.priordict['uniform'].keys():
		# 		Av = (
		# 			(max(self.priordict['uniform']['Av'])-min(self.priordict['uniform']['Av']))*uAv + 
		# 			min(self.priordict['uniform']['Av']))
		# 	elif 'Av' in self.priordict['gaussian'].keys():
		# 		Av = norm.ppf(uAv,loc=self.priordict['gaussian']['Av'][0],scale=self.priordict['gaussian']['Av'][1])
		# 	else:
		# 		Av = (5.0-0.0)*uAv + 0.0
		# 	outdict['Av'] = Av

		# if 'Rv' in upars.keys():
		# 	uRv = upars['Rv']
		# 	if 'Rv' in self.priordict['uniform'].keys():
		# 		Rv = (
		# 			(max(self.priordict['uniform']['Rv'])-min(self.priordict['uniform']['Rv']))*uRv + 
		# 			min(self.priordict['uniform']['Rv']))
		# 	elif 'Rv' in self.priordict['gaussian'].keys():
		# 		Rv = norm.ppf(uRv,loc=self.priordict['gaussian']['Rv'][0],scale=self.priordict['gaussian']['Rv'][1])
		# 	else:
		# 		Rv = (5.0-2.0)*uRv + 2.0
		# 	outdict['Rv'] = Rv

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

