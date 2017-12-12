import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.chebyshev import Chebyshev as T

### NEED TO FIX THIS CLASS ####
class RVcalc(object):
	"""docstring for RVcalc"""
	def __init__(self, arg):
		super(RVcalc, self).__init__()
		self.arg = arg
		

	def rvfit(self,wave,flux,eflux,initpars=None,normspec_bool=False,nnpath=None):
		if initpars == None:
			initpars = [5770.0,4.44,0.0,0.0,0.0,1.0,32000.0]
		init_vrad = 0.0

		fitargs['obs_wave_fit']  = wave
		fitargs['obs_flux_fit']  = flux
		fitargs['obs_eflux_fit'] = eflux
		fitargs['specANNpath']   = nnpath
		runbools = [True,False,normspec_bool]

		from .likelihood import likelihood
		likefn = likelihood(fitargs,runbools)

		args = [initpars,wave,flux,eflux,normspec_bool]
		return minimize_scalar(
			chisq_rv, init_vrad, args=args, 
			bounds=(-500,500),method='bounded',
			options={'disp': 0, 'maxiter': 2000, 'xatol': 1e-10})

	def chisq_rv(self,rv,args):
		initpars = args[0]
		wave = args[1]
		flux = args[2]
		eflux = args[3]
		normspec_bool = args[4]
		initpars[4] = rv
		modwave,modflux = self.genspec(initpars,outwave=wave,normspec_bool=normspec_bool)
		chisq = np.sum([((m-o)**2.0)/(s**2.0) for m,o,s in zip(
			modflux,flux,eflux)])
		return chisq

def polycalc(coef,inwave):
	# define obs wave on normalized scale
	x = inwave - inwave.min()
	x = 2.0*(x/x.max())-1.0
	# build poly coef
	c = np.insert(coef[1:],0,0)
	poly = chebval(x,c)
	epoly = np.exp(coef[0]+poly)
	return epoly

def airtovacuum(inwave):
	"""
		Using the relationship from Ciddor (1996) and transcribed in Shetrone et al. 2015
	"""
	a = 0.0
	b1 = 5.792105E-2
	b2 = 1.67917E-3
	c1 = 238.0185
	c2 = 57.362

	deltawave = a + (b1/(c1-(1.0/inwave**2.0))) + (b2/(c2-(1.0/inwave**2.0)))

	return inwave*(deltawave+1)
