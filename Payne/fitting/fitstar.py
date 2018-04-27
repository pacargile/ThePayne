# #!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from datetime import datetime 

import dynesty

from .fitutils import airtovacuum

def lnprobfn(pars,likeobj,priorobj):
	# first pass pars into priorfn
	lnprior = priorobj.lnpriorfn(pars)

	# check to see if outside of a flat prior
	if lnprior == -np.inf:
		return -np.inf

	lnlike = likeobj.lnlikefn(pars)
	if lnlike == -np.inf:
		return -np.inf
	
	return lnprior + lnlike	

class FitPayne(object):
	"""docstring for FitPayne"""
	def __init__(self,**kwargs):
		from .likelihood import likelihood
		from .prior import prior
		self.prior = prior
		self.likelihood = likelihood
		self.oldnnbool = kwargs.get('oldnn',False)

	def run(self,*args,**kwargs):
		# set verbose
		self.verbose = kwargs.get('verbose',True)

		# check to make sure there is a datadict, can't fit otherwise
		if 'inputdict' in kwargs:
			inputdict = kwargs['inputdict']
		else:
			print('NO USER DEFINED INPUT DICT, NOTHING TO FIT!')
			raise IOError

		# define user defined prior dict
		self.priordict = inputdict.get('priordict',{})

		# define output file
		self.output = inputdict.get('output','Test.dat')

		# determine if there is a sampler dict in input
		self.samplerdict = inputdict.get('sampler',{})

		# initialize a dict for input arguments into the fitter
		self.fitargs = {}

		# set some flags
		self.spec_bool = False
		self.phot_bool = False
		self.normspec_bool = False

		# determine if input has an observed spectrum
		if 'spec' in inputdict.keys():
			self.spec_bool = True
			# stick observed spectrum into fitargs
			self.fitargs['obs_wave']  = inputdict['spec']['obs_wave']
			self.fitargs['obs_flux']  = inputdict['spec']['obs_flux']
			self.fitargs['obs_eflux'] = inputdict['spec']['obs_eflux']

			# deterimine if user defined spec ANN path
			self.fitargs['specANNpath'] = inputdict.get('specANNpath',None)

			# check to see if user defined some wavelength range for spectrum
			if 'wave_minmax' in inputdict['spec'].keys():
				self.fitargs['wave_minmax'] = inputdict['spec']['wave_minmax']
				wavecond = ((self.fitargs['obs_wave'] >= inputdict['spec']['wave_minmax'][0]) 
					& (self.fitargs['obs_wave'] <= inputdict['spec']['wave_minmax'][1]))
				self.fitargs['obs_wave_fit']  = self.fitargs['obs_wave'][wavecond]
				self.fitargs['obs_flux_fit']  = self.fitargs['obs_flux'][wavecond]
				self.fitargs['obs_eflux_fit'] = self.fitargs['obs_eflux'][wavecond]
			else:
				self.fitargs['obs_wave_fit']  = self.fitargs['obs_wave']
				self.fitargs['obs_flux_fit']  = self.fitargs['obs_flux']
				self.fitargs['obs_eflux_fit'] = self.fitargs['obs_eflux']

			if inputdict['spec'].get('convertair',True):
				# shift data to vacuum to match C3K
				self.fitargs['obs_wave_fit'] = airtovacuum(self.fitargs['obs_wave_fit'])

			# determine if user wants to fit the continuum normalization
			if 'normspec' in inputdict['spec'].keys():
				# check to see if normspec is True
				if inputdict['spec']['normspec']:
					if self.verbose:
						print('... Fitting a Blaze function')
					self.normspec_bool = True
					# check to see if user defined a series of polynomial coef for 
					# blaze function as priors
					if 'blaze_coeff' in inputdict['priordict'].keys():
						self.polyorder = len(inputdict['priordict']['blaze_coeff'])
						self.polycoefarr = inputdict['priordict']['blaze_coeff']
					elif 'polyorder' in inputdict['spec'].keys():
						# check to see if user defined blaze poly order
						self.polyorder = inputdict['spec']['polyorder']
						self.polycoefarr = ([[0.0,0.5] for _ in range(self.polyorder)])
					else:
						# by default use a 3rd order poly
						self.polyorder = 3
						self.polycoefarr = ([[0.0,0.5] for _ in range(self.polyorder)])
					if self.verbose:
						print('... Fitting a Blaze function with polyoder: {0}'.format(self.polyorder))
					self.fitargs['norm_polyorder'] = self.polyorder
					# re-scale the wavelength array from -1 to 1 for the Cheb poly
					self.fitargs['obs_wave_fit_norm'] = (
						self.fitargs['obs_wave_fit'] - self.fitargs['obs_wave_fit'].min())
					self.fitargs['obs_wave_fit_norm'] = (2.0 * 
						(self.fitargs['obs_wave_fit_norm']/self.fitargs['obs_wave_fit_norm'].max())-1.0)		

		if 'phot' in inputdict.keys():
			# input filters
			photfilters = inputdict['phot'].keys()

			# input ANN path for photometry
			photANNpath = inputdict.get('photANNpath',None)

			# initialize the photometric ANN, first args are the filters, second is the 
			# path to the ANN HDF5 file.
			self.fitargs['photANNpath'] = photANNpath
			self.phot_bool = True

			# stick phot into fitargs
			self.fitargs['obs_phot'] = {kk:inputdict['phot'][kk] for kk in inputdict['phot'].keys()}

		# run the fitter
		return self({
			'fitargs':self.fitargs,
			'sampler':self.samplerdict,
			'priordict':self.priordict,
			'runbools':[self.spec_bool,self.phot_bool,self.normspec_bool,self.oldnnbool]
			})

	def __call__(self,indicts):
		'''
		call instance so that run_dynesty can be called with multiprocessing
		and still have all of the class instance variables
		'''
		return self.run_dynesty(indicts)
		

	def run_dynesty(self,indicts):
		# split indicts
		fitargs = indicts['fitargs']
		priordict = indicts['priordict']
		samplerdict = indicts['sampler']
		runbools = indicts['runbools']

		# determine the number of dims
		if self.spec_bool:
			self.ndim = 7

		if self.phot_bool:
			if self.spec_bool:
				self.ndim = 10
			else:
				self.ndim = 6

		if self.normspec_bool:
			if self.phot_bool:
				self.ndim = 10+self.polyorder+1
			else:
				self.ndim = 7+self.polyorder+1

		# initialize the output file
		self._initoutput()

		# initialize the prior class
		self.priorobj = self.prior(priordict,runbools)

		# initialize the likelihood class
		self.likeobj = self.likelihood(fitargs,runbools)

		# run sampler and return sampler object
		return self.runsampler(samplerdict)

	def _initoutput(self):
		# init output file
		self.outff = open(self.output,'w')
		self.outff.write('Iter ')
		if self.spec_bool:
			self.outff.write('Teff logg FeH aFe Vrad Vrot Inst_R ')

			if self.normspec_bool:
				for ii in range(self.polyorder+1):
					self.outff.write('pc_{0} '.format(ii))

		if self.phot_bool:
			if not self.spec_bool:
				self.outff.write('Teff logg FeH ')

			self.outff.write('logR Dist Av ')
		self.outff.write('log(lk) log(vol) log(wt) h nc log(z) delta(log(z))')
		self.outff.write('\n')

	def runsampler(self,samplerdict):
		# pull out user defined sampler variables
		npoints = samplerdict.get('npoints',200)
		samplertype = samplerdict.get('samplertype','multi')
		bootstrap = samplerdict.get('bootstrap',0)
		update_interval = samplerdict.get('update_interval',0.6)
		samplemethod = samplerdict.get('samplemethod','unif')
		delta_logz_final = samplerdict.get('delta_logz_final',0.01)
		flushnum = samplerdict.get('flushnum',10)
		maxiter = samplerdict.get('maxiter',sys.maxint)

		# set start time
		starttime = datetime.now()
		if self.verbose:
			print(
				'Start Dynesty w/ {0} number of samples, Ndim = {1}, and w/ stopping criteria of dlog(z) = {2}: {3}'.format(
					npoints,self.ndim,delta_logz_final,starttime))
		sys.stdout.flush()

		# initialize sampler object
		dy_sampler = dynesty.NestedSampler(
			lnprobfn,
			self.priorobj.priortrans,
			self.ndim,
			logl_args=[self.likeobj,self.priorobj],
			nlive=npoints,
			bound=samplertype,
			sample=samplemethod,
			update_interval=update_interval,
			bootstrap=bootstrap,
			)

		sys.stdout.flush()

		ncall = 0
		nit = 0

		iter_starttime = datetime.now()
		deltaitertime_arr = []

		# start sampling
		for it, results in enumerate(dy_sampler.sample(dlogz=delta_logz_final)):
			(worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar,
				h, nc, worst_it, propidx, propiter, eff, delta_logz) = results			

			self.outff.write('{0} '.format(it))
			self.outff.write(' '.join([str(q) for q in vstar]))
			self.outff.write(' {0} {1} {2} {3} {4} {5} {6} '.format(
				loglstar,logvol,logwt,h,nc,logz,delta_logz))
			self.outff.write('\n')

			ncall += nc
			nit = it

			deltaitertime_arr.append((datetime.now()-iter_starttime).total_seconds()/float(nc))
			iter_starttime = datetime.now()

			if ((it%flushnum) == 0) or (it == maxiter):
				self.outff.flush()

				if self.verbose:
					# format/output results
					if logz < -1e6:
						logz = -np.inf
					if delta_logz > 1e6:
						delta_logz = np.inf
					if logzvar >= 0.:
						logzerr = np.sqrt(logzvar)
					else:
						logzerr = np.nan
					if logzerr > 1e6:
						logzerr = np.inf
						
					sys.stdout.write("\riter: {0:d} | nc: {1:d} | ncall: {2:d} | eff(%): {3:6.3f} | "
						"logz: {4:6.3f} +/- {5:6.3f} | dlogz: {6:6.3f} > {7:6.3f}   | mean(time):  {8}  "
						.format(nit, nc, ncall, eff, 
							logz, logzerr, delta_logz, delta_logz_final,np.mean(deltaitertime_arr)))
					sys.stdout.flush()
					deltaitertime_arr = []
			if (it == maxiter):
				break

		# add live points to sampler object
		for it2, results in enumerate(dy_sampler.add_live_points()):
			# split up results
			(worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar,
			h, nc, worst_it, boundidx, bounditer, eff, delta_logz) = results

			self.outff.write('{0} '.format(nit+it2))

			self.outff.write(' '.join([str(q) for q in vstar]))
			self.outff.write(' {0} {1} {2} {3} {4} {5} {6} '.format(
				loglstar,logvol,logwt,h,nc,logz,delta_logz))
			self.outff.write('\n')

			ncall += nc

			ncall += nc

			if self.verbose:
				# format/output results
				if logz < -1e6:
					logz = -np.inf
				if delta_logz > 1e6:
					delta_logz = np.inf
				if logzvar >= 0.:
					logzerr = np.sqrt(logzvar)
				else:
					logzerr = np.nan
				if logzerr > 1e6:
					logzerr = np.inf
				sys.stdout.write("\riter: {:d} | nc: {:d} | ncall: {:d} | eff(%): {:6.3f} | "
					"logz: {:6.3f} +/- {:6.3f} | dlogz: {:6.3f} > {:6.3f}      "
					.format(nit + it2, nc, ncall, eff, 
						logz, logzerr, delta_logz, delta_logz_final))

				sys.stdout.flush()

		self.outff.close()
		sys.stdout.write('\n')

		finishtime = datetime.now()
		if self.verbose:
			print('RUN TIME: {0}'.format(finishtime-starttime))

		return dy_sampler		
