import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.chebyshev import Chebyshev as T
from scipy.interpolate import interp1d
from scipy import constants
speedoflight = constants.c / 1000.0
from scipy.optimize import minimize_scalar, minimize, brute, basinhopping, differential_evolution

### NEED TO FIX THIS CLASS ####
class RVcalc(object):
	def __init__(self, **kwargs):
		super(RVcalc, self).__init__()
		
		self.wave = kwargs.get('inwave',[])
		self.flux = kwargs.get('influx',[])
		self.eflux = kwargs.get('einflux',[])
		self.modflux = kwargs.get('modflux',[])
		self.modwave = kwargs.get('modwave',[])

	def __call__(self):
		init_vrad = 0.0

		args = [self.wave,self.flux,self.eflux,self.modflux,self.modwave]

		outfn = brute(
			self.chisq_rv,
			(slice(-700,700,0.1),),
			)
		return outfn

		# return minimize(
		# 	self.chisq_rv,
		# 	init_vrad,
		# 	method='Nelder-Mead',
		# 	tol=10E-10,
		# 	options={'maxiter':1E6}
		# 	)

	def chisq_rv(self,rv):
		wave = self.wave
		flux = self.flux
		eflux = self.eflux
		modflux = self.modflux
		modwave = self.modwave

		# adjust model to new rv
		modwave_i = modwave*(1.0+(rv/speedoflight))

		# interpolate ict back to wave so that chi-sq can be computed
		modflux_i = interp1d(modwave_i,modflux,kind='linear',bounds_error=False,fill_value=1.0)(wave)

		chisq = np.sum([((m-o)**2.0)/(s**2.0) for m,o,s in zip(
			modflux_i,flux,eflux)])
		return chisq

class PCcalc(object):
	def __init__(self, **kwargs):
		super(PCcalc, self).__init__()
		
		self.wave = kwargs.get('inwave',[])
		self.flux = kwargs.get('influx',[])
		self.eflux = kwargs.get('einflux',[])
		self.modflux = kwargs.get('modflux',[])
		self.modwave = kwargs.get('modwave',[])
		self.numpoly = kwargs.get('numpoly',4)

	def __call__(self):
		init_pc = [1.0] + [0.0 for _ in range(self.numpoly-1)]
		args = [self.wave,self.flux,self.eflux,self.modflux,self.modwave]

		return minimize(
			self.chisq_pc,
			init_pc,
			method='Nelder-Mead',
			tol=10E-15,
			options={'maxiter':1E4}
			)

	def chisq_pc(self,pc):
		wave = self.wave
		flux = self.flux
		eflux = self.eflux
		modflux = self.modflux
		modwave = self.modwave

		polymod = polycalc(pc,wave)

		# interpolate ict back to wave so that chi-sq can be computed
		flux_i = flux / interp1d(modwave,modflux,kind='linear',bounds_error=False,fill_value=1.0)(wave)

		chisq = np.sum([((m-o)**2.0)/(s**2.0) for m,o,s in zip(
			polymod,flux_i,eflux)])
		return chisq


def polycalc(coef,inwave):
	# define obs wave on normalized scale
	x = inwave - inwave.min()
	x = 2.0*(x/x.max())-1.0
	# build poly coef
	# c = np.insert(coef[1:],0,0)
	poly = chebval(x,coef)
	# epoly = np.exp(coef[0]+poly)
	# epoly = coef[0]+poly
	return poly

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