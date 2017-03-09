# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import h5py
from scipy import constants
speedoflight = constants.c / 1000.0
from scipy.interpolate import UnivariateSpline

from ..predict.predictspec import PaynePredict

class TestSpec(object):
	"""
	Class for testing a Payne-learned NN using a testing dataset 
	different from training spectra
	"""
	def __init__(self, NNfilename):
		# user inputed neural-net output file
		self.NNfilename = NNfilename

		# initialize the Payne Predictor
		self.PP = PaynePredict(self.NNfilename)


	def pulltestspectra(self,testnum,**kwargs):
		'''
		Function to setup the testing spectra sample

		:params testnum
			Number of spectra randomly drawn for the 
			testing sample

		:params label (optional):
			kwarg defined as labelname=[min, max]
			This constrains the spectra to only 
			be drawn from a given range of labels

		:returns spectra:
			Structured array: wave, spectra1, spectra2, spectra3, ...
			where spectrai is a flux array for ith spectrum. wave is the
			wavelength array in nm.

		: returns labels:
			Array of labels for the individual drawn spectra
		'''

		self.testnum = testnum

		if 'Teff' in kwargs:
			Teffrange = kwargs['Teff']
		else:
			Teffrange = [3500.0,10000.0]

		if 'logg' in kwargs:
			loggrange = kwargs['logg']
		else:
			loggrange = [-1.0,5.0]

		if 'FeH' in kwargs:
			fehrange = kwargs['FeH']
		else:
			fehrange = [-2.0,0.5]

		if 'resolution' in kwargs:
			resolution = kwargs['resolution']
		else:
			resolution = None

		# define the [Fe/H] array, this is the values that the MIST 
		# and C3K grids are built
		FeHarr = [-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5]

		# define aliases for the MIST isochrones and C3K/CKC files
		# MISTpath = '/n/regal/conroy_lab/pac/ThePayne/models/MIST/'
		# C3Kpath  = '/n/regal/conroy_lab/pac/ThePayne/models/CKC/'

		MISTpath = os.path.dirname(Payne.__file__[:-19]+'/data/')
		C3Kpath  = os.path.dirname(Payne.__file__[:-19]+'/data/')

		# load MIST models
		MIST = h5py.File(MISTpath+'MIST_full.h5','r')
		MIST_EAF = np.array(MIST['EAF'])
		MIST_BSP = np.array(MIST['BSP'])

		# parse down the MIST models to just be EEP = 202-605
		EEPcond = (MIST_EAF['EEP'] > 202) & (MIST_EAF['EEP'] < 605)
		EEPcond = np.array(EEPcond,dtype=bool)
		MIST_EAF = MIST_EAF[EEPcond]
		MIST_BSP = MIST_BSP[EEPcond]

		# create a dictionary for the C3K models and populate it for different
		# metallicities
		C3K = {}

		for mm in FeHarr:
			C3K[mm] = h5py.File(C3Kpath+'ckc_feh={0:+4.2f}.full.h5'.format(mm),'r')

		# randomly select num number of MIST isochrone grid points, currently only 
		# using dwarfs, subgiants, and giants (EEP = 202-605)

		labels = []
		spectra = []

		for ii in range(testnum):
			while True:
				# first randomly draw a [Fe/H]
				while True:
					FeH_i = np.random.choice(FeHarr)

					# check to make sure FeH_i isn't in user defined 
					# [Fe/H] limits
					if (FeH_i >= fehrange[0]) & (FeH_i <= fehrange[1]):
						break

				# select the C3K spectra at that [Fe/H]
				C3K_i = C3K[FeH_i]

				# store a wavelength array as an instance, all of C3K has 
				# the same wavelength sampling
				if ii == 0:
					wavelength_i = np.array(C3K_i['wavelengths'])
					if resolution != None:
						# define new wavelength array with 3*resolution element sampling
						wavelength_o = []
						i = 1
						while True:
							wave_i = self.waverange[0]*(1.0 + 1.0/(3.0*resolution))**(i-1.0)
							if wave_i <= self.waverange[1]:
								wavelength_o.append(wave_i)
								i += 1
							else:
								break
						wavelength_o = np.array(wavelength_o)
					else:
						wavecond = (wavelength_i >= self.waverange[0]) & (wavelength_i <= self.waverange[1])
						wavecond = np.array(wavecond,dtype=bool)
						wavelength_o = wavelength_i[wavecond]


				# select the range of MIST isochrones with that [Fe/H]
				FeHcond = (MIST_EAF['initial_[Fe/H]'] == FeH_i)
				MIST_BSP_i = MIST_BSP[np.array(FeHcond,dtype=bool)]

				while True:
					# randomly select a EEP, log(age) combination
					MISTsel = np.random.randint(0,len(MIST_BSP_i))

					# get MIST Teff and log(g) for this selection
					logt_MIST,logg_MIST = MIST_BSP_i['log_Teff'][MISTsel], MIST_BSP_i['log_g'][MISTsel]

					# do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
					C3Kpars = np.array(C3K_i['parameters'])

					# check to make sure MIST log(g) and log(Teff) have a spectrum in the C3K grid
					# if not draw again
					if (
						(logt_MIST >= np.log10(Teffrange[0])) and (logt_MIST <= np.log10(Teffrange[1])) and
						(logg_MIST >= loggrange[0]) and (logg_MIST <= loggrange[1])
						):
						break
				C3KNN = NearestNDInterpolator(
					np.array([C3Kpars['logt'],C3Kpars['logg']]).T,range(0,len(C3Kpars))
					)((logt_MIST,logg_MIST))

				# determine the labels for the selected C3K spectrum
				label_i = list(C3Kpars[C3KNN])[:-1]

				# calculate the normalized spectrum
				spectra_i = C3K_i['spectra'][C3KNN]/C3K_i['continuua'][C3KNN]

				# check to see if label_i in labels, or spectra_i is nan's
				# if so, then skip the append and go to next step in while loop
				# do this before the smoothing to reduce run time
				if (label_i in labels) or (np.any(np.isnan(spectra_i))):
					continue

				# check to make sure label_i is not in the NN labels
				if (label_i in self.PP.NN['labels']):
					continue

				# if user defined resolution to train at, the smooth C3K to that resolution
				if resolution != None:
					spectra_i = self.smoothspec(wavelength_i,spectra_i,resolution,
						outwave=wavelength_o,smoothtype='R',fftsmooth=True)
				else:
					spectra_i = spectra_i[wavecond]

				# check to see if labels are already in training set, if not store labels/spectrum
				if (label_i not in labels) and (not np.any(np.isnan(spectra_i))):
					labels.append(label_i)
					spectra.append(spectra_i)
					break

		return np.array(spectra), np.array(labels), wavelength_o


	def runtest(self,**kwargs):
		'''
		Function to run the testing on an already trained network
		'''
		if 'testnum' in kwargs:
			testnum = kwargs['testnum']
		else:
			# test with equal number of spectra as training
			testnum = len(self.PP.NN['labels'])

		if 'resolution' in kwargs:
			resolution = kwargs['resolution']
		else:
			# default resolution is the native resolution of NN
			resolution = self.PP.NN['resolution']

		self.spectra_test,self.labels_test,self.wavelength_test = self.pulltestspectra(
			testnum,resolution=resolution)

		# generate predicted spectra at each of the testing spectra labels
		testspecdict = {}
		testspecdict['WAVE'] = self.wavelength_test
		for ii,pars,testspec in zip(range(len(self.labels_test)),self.labels_test,self.spectra_test):
			modwave_i,modflux_i = self.PP.getspec(
				Teff=pars[0],logg=pars[1],feh=pars[2],rad_vel=0.0,rot_vel=0.0,
				inst_R=resolution)

			testspecdict[ii] = {'test':testspec,'train':modflux_i}
		return testspecdict