# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import h5py
from scipy import constants
speedoflight = constants.c / 1000.0
from scipy.interpolate import UnivariateSpline,NearestNDInterpolator

from ..predict import predictspec

from numpy.random import default_rng
rng = default_rng()

import matplotlib
matplotlib.use('AGG')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class TestSpec(object):
	"""
	Class for testing a Payne-learned NN using a testing dataset 
	different from training spectra
	"""
	def __init__(self, NNfilename):
		# user inputed neural-net output file
		self.NNfilename = NNfilename

		# type of NN
		self.nntype = kwargs.get('NNtype','LinNet')

		# initialize the Payne Predictor
		self.NN = predictspec.ANN(nnpath=self.NNfilename,NNtype=self.nntype,verbose=False)

		# define wavelength range of trained network
		self.waverange = [self.NN['wavelength'].min(),self.NN['wavelength'].max()]

		# default resolution is the native resolution of NN
		self.resolution = float(np.array(self.NN['resolution']))


	def runtest(self,**kwargs):
		'''
		Function to run the testing on an already trained network
		'''
		if 'testnum' in kwargs:
			testnum = kwargs['testnum']
			ind = rng.integers(low=0,high=len(self.NN['testlabels']),size=testnum)
			testpred = self.NN['testpred'][ind]
			testlabels = self.NN['testlabels'][ind]
		else:
			# test with equal number of spectra as training
			testnum = len(self.NN['testlabels'])
			testpred = self.NN['testpred'][:]
			testlabels = self.NN['testlabels'][:]


		# make NN predictions for all test labels
		nnpred = np.array([NN.eval(pars) for pars in testlabels])

		output = kwargs.get('output','./test.pdf')

		# initialize PDF file
		with PdfPages(output) as pdf:

			# MAD histograms binned by pars
			fig,ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True)

			

			pdf.savefig(fig)
			plt.close(fig)

			# build MAD versus lambda plots
			fig,ax = plt.subplots(nrows=2,ncols=1,constrained_layout=True)

			pdf.savefig(fig)
			plt.close(fig)

			# build MAD versus lambda binned by pars 2x2
			fig,ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True)

			pdf.savefig(fig)
			plt.close(fig)

			# build MAD map for 4 different pars
			fig,ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True)

			pdf.savefig(fig)
			plt.close(fig)

