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

		# define wavelengths
		self.wave = self.NN['wavelength'][:]

		# default resolution is the native resolution of NN
		self.resolution = float(np.array(self.NN['resolution']))


	def runtest(self,**kwargs):
		'''
		Function to run the testing on an already trained network
		'''
		# output plot file
		output = kwargs.get('output','./test.pdf')

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

		# make residual array
		modres = np.array([np.abs(x-y) for x,y in zip(testpred,nnpred)])

		# initialize PDF file
		with PdfPages(output) as pdf:

			# MAD histograms binned by pars
			fig,ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True)

			# teff binned
			ind = testlabels['teff'] > 7000.0
			ax[0,0].hist(np.median(modres[ind]),bins=25,c='C0')
			ind = (testlabels['teff'] <= 7000.0) & (testlabels['teff'] > 6000.0)
			ax[0,0].hist(np.median(modres[ind]),bins=25,c='C3')
			ind = (testlabels['teff'] <= 5500.0) & (testlabels['teff'] > 4500.0)
			ax[0,0].hist(np.median(modres[ind]),bins=25,c='C4')
			ind = (testlabels['teff'] <= 4500.0) 
			ax[0,0].hist(np.median(modres[ind]),bins=25,c='C5')

			# logg binned
			ind = testlabels['logg'] > 4.5
			ax[0,1].hist(np.median(modres[ind]),bins=25,c='C0')
			ind = (testlabels['logg'] <= 4.5) & (testlabels['logg'] > 4.0)
			ax[0,1].hist(np.median(modres[ind]),bins=25,c='C3')
			ind = (testlabels['logg'] <= 4.0) & (testlabels['logg'] > 3.5)
			ax[0,1].hist(np.median(modres[ind]),bins=25,c='C4')
			ind = (testlabels['logg'] <= 3.5) 
			ax[0,1].hist(np.median(modres[ind]),bins=25,c='C5')

			# feh binned
			ind = testlabels['feh'] > 0.25
			ax[1,0].hist(np.median(modres[ind]),bins=25,c='C0')
			ind = (testlabels['feh'] <= 0.25) & (testlabels['feh'] > -0.5)
			ax[1,0].hist(np.median(modres[ind]),bins=25,c='C3')
			ind = (testlabels['feh'] <= -0.5) & (testlabels['feh'] > -1.5)
			ax[1,0].hist(np.median(modres[ind]),bins=25,c='C4')
			ind = (testlabels['feh'] <= -1.5) 
			ax[1,0].hist(np.median(modres[ind]),bins=25,c='C5')

			# afe binned
			ind = testlabels['afe'] > 0.4
			ax[1,1].hist(np.median(modres[ind]),bins=25,c='C0')
			ind = (testlabels['afe'] <= 0.4) & (testlabels['afe'] > 0.2)
			ax[1,1].hist(np.median(modres[ind]),bins=25,c='C3')
			ind = (testlabels['afe'] <= 0.2) & (testlabels['afe'] > 0.0)
			ax[1,1].hist(np.median(modres[ind]),bins=25,c='C4')
			ind = (testlabels['afe'] <= 0.0) 
			ax[1,1].hist(np.median(modres[ind]),bins=25,c='C5')

			ax[1,0].set_xlabel('median MAD per spectrum')
			ax[1,1].set_xlabel('median MAD per spectrum')

			pdf.savefig(fig)
			plt.close(fig)

			# build MAD versus lambda plots
			fig,ax = plt.subplots(nrows=2,ncols=1,constrained_layout=True)

			ax[0].scatter(self,wave,np.median(modres,axis=0),marker='o')
			ax[1].hist(np.median(modres,axis=0),bins=25,cumulative=True,density=True)

			ax[0].set_xlabel(r'$\lambda')
			ax[0].set_xlabel('median MAD @ pixel')

			ax[1].set_xlabel('MAD')
			ax[1].set_ylabel('CDF')

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

