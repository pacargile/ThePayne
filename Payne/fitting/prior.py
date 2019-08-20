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
		self.priordict['exp'] = {}
		self.priordict['texp'] = {}
		self.priordict['loguniform'] = {}


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
					if ii == 'pv_uniform':
						self.priordict['uniform'][kk] = inpriordict[kk]['pv_uniform']
					elif ii == 'pv_gaussian':
						self.priordict['gaussian'][kk] = inpriordict[kk]['pv_gaussian']
					elif ii == 'pv_tgaussian':
						self.priordict['tgaussian'][kk] = inpriordict[kk]['pv_tgaussian']
					elif ii == 'pv_exp':
						self.priordict['exp'][kk] = inpriordict[kk]['pv_exp']
					elif ii == 'pv_texp':
						self.priordict['texp'][kk] = inpriordict[kk]['pv_texp']
					elif ii == 'pv_loguniform':
						self.priordict['loguniform'][kk] = inpriordict[kk]['pv_loguniform']
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
		self.carbon_bool = runbools[5]

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
		self.defaultpars['CarbonScale'] = [0.0,2.0]

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
	
		# calcuate transformation from prior volume to parameter for all modeled parameters

		outdict = {}

		for namepar in ['Teff','log(g)','[Fe/H]','[a/Fe]','Vrad','Vrot','Inst_R','CarbonScale']:
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
					if par_i == np.inf:
						par_i = self.priordict['tgaussian'][namepar][1]
				elif namepar in self.priordict['exp'].keys():
					par_i = expon.ppf(upars_i,loc=self.priordict['exp'][namepar][0],scale=self.priordict['exp'][namepar][1])
				else:
					par_i = (self.defaultpars[namepar][1]-self.defaultpars[namepar][0])*upars_i + self.defaultpars[namepar][0]

				outdict[namepar] = par_i

		# if fitting a blaze function, do transformation for polycoef
		pcarr = [x_i for x_i in upars.keys() if 'pc' in x_i]
		if len(pcarr) > 0:
			for pc_i in pcarr:
				if pc_i == 'pc_0':
					uspec_scale = upars['pc_0']
					outdict['pc_0'] = (1.25 - 0.75)*uspec_scale + 0.75
				else:
					pcind = int(pc_i.split('_')[-1])
					pcmax = self.polycoefarr[pcind][0]+3.0*self.polycoefarr[pcind][1]
					pcmin = self.polycoefarr[pcind][0]-3.0*self.polycoefarr[pcind][1]
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
						if par_i == np.inf:
							par_i = self.priordict['tgaussian'][namepar][1]
					elif namepar in self.priordict['exp'].keys():
						par_i = expon.ppf(upars_i,loc=self.priordict['exp'][namepar][0],scale=self.priordict['exp'][namepar][1])
					else:
						par_i = (self.defaultpars[namepar][1]-self.defaultpars[namepar][0])*upars_i + self.defaultpars[namepar][0]

					outdict[namepar] = par_i
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

				elif namepar in self.priordict['exp'].keys():
					par_i = expon.ppf(upars_i,loc=self.priordict['exp'][namepar][0],scale=self.priordict['exp'][namepar][1])

				elif namepar in self.priordict['tgaussian'].keys():
					a = (self.priordict['tgaussian'][namepar][0] - self.priordict['tgaussian'][namepar][2]) / self.priordict['tgaussian'][namepar][3]
					b = (self.priordict['tgaussian'][namepar][1] - self.priordict['tgaussian'][namepar][2]) / self.priordict['tgaussian'][namepar][3]
					par_i = truncnorm.ppf(upars_i,a,b,loc=self.priordict['tgaussian'][namepar][2],scale=self.priordict['tgaussian'][namepar][3])
					if par_i == np.inf:
						par_i = self.priordict['tgaussian'][namepar][1]

				elif namepar in self.priordict['texp'].keys():
					a = (self.priordict['texp'][namepar][0] - self.priordict['texp'][namepar][2]) / self.priordict['texp'][namepar][3]
					b = (self.priordict['texp'][namepar][1] - self.priordict['texp'][namepar][2]) / self.priordict['texp'][namepar][3]
					par_i = truncexpon.ppf(upars_i,a,b,loc=self.priordict['texp'][namepar][2],scale=self.priordict['texp'][namepar][3])
					if par_i == np.inf:
						par_i = self.priordict['texp'][namepar][1]

				elif namepar in self.priordict['loguniform'].keys():
					par_i = reciprocal.ppf(upars_i, self.priordict['loguniform'][namepar][0], self.priordict['loguniform'][namepar][1])

				else:
					par_i = (self.defaultpars[namepar][1]-self.defaultpars[namepar][0])*upars_i + self.defaultpars[namepar][0]

				outdict[namepar] = par_i

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
				if kk in ['Teff','log(g)','[Fe/H]','[a/Fe]','Vrad','Vrot','Inst_R','CarbonScale']:
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

		# # if fitting a blaze function, then check for additional priors
		# if self.normspec_bool:
		# 	for pp in self.fitpars_i:
		# 		if pp[:2] == 'pc_':
		# 			lnprior += -0.5 * ((parsdict[pp]/self.polycoefarr[kk][1])**2.0)

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
			logM = pardict['log(g)'] + 2.0 * pardict['log(R)']
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

