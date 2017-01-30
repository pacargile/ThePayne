# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from scipy import constants
speedoflight = constants.c / 1000.0
from scipy.optimize import minimize

class SpecMinimize(object):
	"""docstring for SpecMinimize"""
	def __init__(self,chi2fn, args, opts, method='powell'):
		super(SpecMinimize, self).__init__()
		self.chi2fn = chi2fn
        self.method = method
        self.opts = opts
        self.args = args

    def run(self,pinit):
		output = minimize(self.chi2fn, pinit, args=self.args,
			method=self.method, options=self.opts)
		return output

class SpecEmcee(object):
	"""docstring for SpecEmcee"""
	def __init__(self, arg):
		super(SpecEmcee, self).__init__()
		self.arg = arg

