# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import h5py
from multiprocessing import Pool
from scipy.interpolate import NearestNDInterpolator
import os,sys
from itertools import imap

from ..utils.smoothing import smoothspec

class pullspectra(object):
	def __init__(self,):
		# define the [Fe/H] array, this is the values that the MIST 
		# and C3K grids are built
		self.FeHarr = [-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5]

		# define the [alpha/Fe] array
		self.alphaarr = [0.0,0.2,0.4]

		# define aliases for the MIST isochrones and C3K/CKC files
		currentpath = __file__
		if currentpath[-1] == 'c':
			removeind = -27
		else:
			removeind = -26
		self.MISTpath = os.path.dirname(__file__[:removeind]+'data/MIST/')
		self.C3Kpath  = os.path.dirname(__file__[:removeind]+'data/C3K/')

		# load MIST models
		MIST = h5py.File(self.MISTpath+'/MIST_full.h5','r')
		MIST_MOD = np.array(MIST['MODPARS'])
		MIST_STA = np.array(MIST['STARPARS'])

		# parse down the MIST models to just be EEP = 202-605
		EEPcond = (MIST_MOD['EEP'] > 202) & (MIST_MOD['EEP'] < 605)
		EEPcond = np.array(EEPcond,dtype=bool)
		self.MIST_MOD = MIST_MOD[EEPcond]
		self.MIST_STA = MIST_STA[EEPcond]
	
	def __call__(self,num,**kwargs):
		'''
		Randomly draw 2*num spectra from C3K based on 
		the MIST isochrones.
		
		:params num:
			Number of spectra randomly drawn for training

		:params label (optional):
			kwarg defined as labelname=[min, max]
			This constrains the spectra to only 
			be drawn from a given range of labels

		:params excludelabels (optional):
			kwarg defined as array of labels
			that should not be included in 
			output sample of spectra. Useful 
			for when defining validation and 
			testing spectra

		: params waverange (optional):
			kwarg used to set wavelength range
			of output spectra

		:returns spectra:
			Structured array: wave, spectra1, spectra2, spectra3, ...
			where spectrai is a flux array for ith spectrum. wave is the
			wavelength array in nm.

		: returns labels:
			Array of labels for the individual drawn spectra

		'''

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

		if 'aFe' in kwargs:
			aFerange = kwargs['aFe']
		else:
			aFerange = [0.0,0.4]

		if 'resolution' in kwargs:
			resolution = kwargs['resolution']
		else:
			resolution = None

		if 'excludelabels' in kwargs:
			excludelabels = kwargs['excludelabels'].T.tolist()
		else:
			excludelabels = []

		if 'waverange' in kwargs:
			waverange = kwargs['waverange']
		else:
			# default is just the MgB triplet 
			waverange = [5150.0,5200.0]


		# create a dictionary for the C3K models and populate it for different
		# metallicities
		C3K = {}
		for aa in self.alphaarr:
			C3K[aa] = {}
			for mm in self.FeHarr:
				C3K[aa][mm] = h5py.File(
					self.C3Kpath+'/c3k_v1.3_feh{0:+4.2f}_afe{1:+3.1f}.full.h5'.format(mm,aa),
					'r')

		# randomly select num number of MIST isochrone grid points, currently only 
		# using dwarfs, subgiants, and giants (EEP = 202-605)

		labels = []
		spectra = []
		wavelength_o = []

		for ii in range(num):
			while True:
				# first randomly draw a [Fe/H]
				while True:
					FeH_i = np.random.choice(self.FeHarr)

					# check to make sure FeH_i isn't in user defined 
					# [Fe/H] limits
					if (FeH_i >= fehrange[0]) & (FeH_i <= fehrange[1]):
						break


				# then draw an alpha abundance
				alpha_i = np.random.choice(self.alphaarr)

				# select the C3K spectra at that [Fe/H] and [alpha/Fe]
				C3K_i = C3K[alpha_i][FeH_i]

				# select the range of MIST isochrones with that [Fe/H]
				FeHcond = (self.MIST_MOD['initial_[Fe/H]'] == FeH_i)
				MIST_STA_i = self.MIST_STA[np.array(FeHcond,dtype=bool)]

				while True:
					# randomly select a EEP, log(age) combination
					MISTsel = np.random.randint(0,len(MIST_STA_i))

					# get MIST Teff and log(g) for this selection
					logt_MIST,logg_MIST = MIST_STA_i['log_Teff'][MISTsel], MIST_STA_i['log_g'][MISTsel]

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
				label_i = list(C3Kpars[C3KNN])

				# check to see if user defined labels to exclude, if so
				# continue on to the next iteration
				if list(label_i) in excludelabels:
					continue

				# calculate the normalized spectrum
				spectra_i = C3K_i['spectra'][C3KNN]/C3K_i['continuua'][C3KNN]

				# check to see if label_i in labels, or spectra_i is nan's
				# if so, then skip the append and go to next step in while loop
				# do this before the smoothing to reduce run time
				if (label_i in labels) or (np.any(np.isnan(spectra_i))):
					continue

				# store a wavelength array as an instance, all of C3K has 
				# the same wavelength sampling
				if wavelength_o == []:
					wavelength_i = np.array(C3K_i['wavelengths'])
					if resolution != None:
						# define new wavelength array with 3*resolution element sampling
						i = 1
						while True:
							wave_i = waverange[0]*(1.0 + 1.0/(3.0*resolution))**(i-1.0)
							if wave_i <= waverange[1]:
								wavelength_o.append(wave_i)
								i += 1
							else:
								break
						wavelength_o = np.array(wavelength_o)
					else:
						wavecond = (wavelength_i >= waverange[0]) & (wavelength_i <= waverange[1])
						wavecond = np.array(wavecond,dtype=bool)
						wavelength_o = wavelength_i[wavecond]

				# if user defined resolution to train at, the smooth C3K to that resolution
				if resolution != None:
					spectra_i = self.smoothspecfunc(wavelength_i,spectra_i,resolution,
						outwave=wavelength_o,smoothtype='R',fftsmooth=True)
				else:
					spectra_i = spectra_i[wavecond]

				# check to see if labels are already in training set, if not store labels/spectrum
				if (label_i not in labels) and (not np.any(np.isnan(spectra_i))):
					labels.append(label_i)
					spectra.append(spectra_i)
					break

		return np.array(spectra), np.array(labels), wavelength_o

	def smoothspecfunc(self,wave, spec, sigma, outwave=None, **kwargs):
		outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
		return outspec
