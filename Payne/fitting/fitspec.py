# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from scipy import constants
speedoflight = constants.c / 1000.0

from ..predict.predictspec import PaynePredict
from .fitter import SpecMinimize,SpecEmcee

class FitSpec(object):
	"""docstring for FitSpec"""
	def __init__(self, arg):
		super(FitSpec, self).__init__()
		self.arg = arg

	