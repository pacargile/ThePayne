# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys,glob,warnings
import numpy as np
from datetime import datetime

import Payne
from .smoothing import smoothspec

from astropy.table import Table

class carbonmod(object):
	def __init__(self,**kwargs):

		self.respfnpath = kwargs.get('respfpath',Payne.__abspath__+'data/carbon/respfn.fits')
		# load respfn 
		self.respfn = Table.read(self.respfnpath,format='fits')

		# wavelength points the model was created on
		outwave = kwargs.get('outwave',None)

		# smooth respfn to ANN resolution
		inres = kwargs.get('inres',500000.0)

		# smooth respfn to ANN resolution
		outres = kwargs.get('outres',100000.0)

		self.respfnratio = self.smoothspec(
			self.respfn['WAVE'],self.respfn['RATIO'],
			outres,outwave=outwave,smoothtype='R',
			fftsmooth=True,inres=inres)

	def applycarbon(self,influx,CarbonScale):
		# M * [A*(RF-1)+1]
		outflux = influx * (CarbonScale * ((1.0/self.respfnratio)-1.0) + 1.0)
		return outflux

	def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
		outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
		return outspec		