import numpy as np

class prior(object):
	"""docstring for priors"""
	def __init__(self, priordict,runbools):
		super(prior, self).__init__()
		self.priordict = priordict
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

		# if (self.ndim == 7) or (self.ndim == 7+self.polyorder+1):
		# 	Teff,logg,FeH,aFe,radvel,rotvel,inst_R = upars[:7]
		# elif (self.ndim == 10) or (self.ndim == 10+self.polyorder+1):
		# 	Teff,logg,FeH,aFe,radvel,rotvel,inst_R = upars[:7]
		# 	logR,Dist,Av = upars[-3:]
		# else:
		# 	#self.ndim == 6
		# 	Teff,logg,FeH,logR,Dist,Av = upars

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

		if self.phot_bool:
			uphotpars = [Teff,logg,FeH,logR,Dist,Av]
			photPT = self.priortrans_phot(uphotpars)
		else:
			photPT = []

		outpars = specPT + photPT
		return outpars

	def priortrans_spec(self,upars):
		uTeff   = upars[0]
		ulogg   = upars[1]
		uFeH    = upars[2]
		uaFe    = upars[3]
		uradvel = upars[4]
		urotvel = upars[5]
		uinst_R = upars[6]
	
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

		if self.normspec_bool:
			uspec_scale = upars[7]
			upolycoef = upars[8:]

			# spec_scale = (2.0*self.fitargs['obs_flux_fit'].max()-0.0)*uspec_scale + 0.0
			spec_scale = (0.1-0.0)*uspec_scale + 0.0
			outarr.append(spec_scale)

			for ii,upolycoef_i in enumerate(upolycoef):
				pcmax = self.polycoefarr[ii][0]+5.0*self.polycoefarr[ii][1]
				pcmin = self.polycoefarr[ii][0]-5.0*self.polycoefarr[ii][1]

				polycoef_i = (pcmax-pcmin)*upolycoef_i + pcmin

				# polycoef_i = 6.0*self.polycoefarr[kk][1]*upolycoef_i + 3.0*self.polycoefarr[kk][1] - self.polycoefarr[kk][0]
				outarr.append(polycoef_i)

		return outarr

	def priortrans_phot(self,upars):

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


	def lnprior_spec(self,pars,verbose=True):
		lnprior = 0.0

		Teff = pars[0]
		logg = pars[1]
		FeH  = pars[2]
		aFe = pars[3]
		radvel = pars[4]
		rotvel = pars[5]
		inst_R = pars[6]

		# check to make sure pars are in grid used to train NN
		if (Teff < 10.0**self.GM.PP.NN['x_min'][0]) | (Teff > 10.0**self.GM.PP.NN['x_max'][0]):
			if verbose:
				print('Hit Teff Prior Bounds!: {0}'.format(Teff))
			return -np.inf
		if (logg < self.GM.PP.NN['x_min'][1]) | (logg > self.GM.PP.NN['x_max'][1]):
			if verbose:
				print('Hit log(g) Prior Bounds!: {0}'.format(logg))
			return -np.inf
		if (FeH < self.GM.PP.NN['x_min'][2])  | (FeH > self.GM.PP.NN['x_max'][2]):
			if verbose:
				print('Hit [Fe/H] Prior Bounds!: {0}'.format(FeH))
			return -np.inf
		if (aFe < self.GM.PP.NN['x_min'][3])  | (aFe > self.GM.PP.NN['x_max'][3]):
			if verbose:
				print('Hit [a/Fe] Prior Bounds!: {0}'.format(aFe))
			return -np.inf
		if (np.abs(radvel) > 400.0):
			if verbose:
				print('Hit Vrad Prior Bounds!: {0}'.format(radvel))
			return -np.inf
		if (rotvel <= 0.0) | (rotvel > 300.0):
			if verbose:
				print('Hit Vrot Prior Bounds!: {0}'.format(rotvel))
			return -np.inf
		if (inst_R <= 10000) | (inst_R > 42000.0):
			if verbose:
				print('Hit Instr_R Prior Bounds!: {0}'.format(inst_R))
			return -np.inf

		if self.normspec_bool:
			spec_scale = pars[7]
			polycoef = pars[8:]

			# for pp in polycoef:
			# 	lnprior += -np.log(np.abs(pp))

			# for kk,pp in enumerate(polycoef):
			# 	lnprior += -0.5 * ((pp-self.polycoefarr[kk][0])**2.0) / ((0.1*self.polycoefarr[kk][0])**2.0)

		# lnprior += -0.5 * ((inst_R-42000.0)**2.0)/(1000.0**2.0)
		# lnprior += -0.5 *((logg-4.16)**2.0)/(0.1**2.0)

		# lnprior += (
		# 	- 0.5*((Teff-5770.0)**2.0)/(10.0**2.0) 
		# 	- 0.5*((logg-4.44)**2.0)/(0.01**2.0) 
		# 	- 0.5*(FeH**2.0)/(0.01**2.0) 
		# 	- 0.5*(aFe**2.0)/(0.01**2.0)
		# 	)

		return lnprior

	def lnprior_phot(self,pars,verbose=True):
		lnprior = 0.0

		Teff = pars[0]
		logg = pars[1]
		FeH  = pars[2]
		logR = pars[3]
		Dist = pars[4]
		Av = pars[5]

		if not self.spec_bool:
			if (Teff < 2500.0) | (Teff > 30000.0):
				if verbose:
					print('Hit phot Teff Prior Bounds!: {0}'.format(Teff))
				return -np.inf
			if (logg < -1.0) | (logg > 6.0):
				if verbose:
					print('Hit phot log(g) Prior Bounds!: {0}'.format(logg))
				return -np.inf
			if (FeH < -4.0) | (FeH > 0.5):
				if verbose:
					print('Hit phot FeH Prior Bounds!: {0}'.format(FeH))
				return -np.inf

		if (logR > 3.0) | (logR < -2.0):
			if verbose:
				print('Hit logR Prior Bounds!: {0}'.format(logR))
			return -np.inf

		if (Dist >= 1000.0) | (Dist < 200.0):
			if verbose:
				print('Hit Dist Prior Bounds!: {0}'.format(Dist))
			return -np.inf

		if (Av < 0.0) | (Av > 3.0):
			if verbose:
				print('Hit Av Prior Bounds!: {0}'.format(Av))
			return -np.inf
		# lnprior += -1.0*(Av/0.1)
		# lnprior += -0.5 * ((Dist-10.0)**2.0)/(0.1**2.0)		
		# lnprior += -0.5 * ( ( (1000.0/Dist) - 2.04)**2.0 ) / (0.3**2.0)

		return lnprior