# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import beta
import os,sys

import Payne
from .smoothing import smoothspec

class pullspectra(object):
	def __init__(self,**kwargs):
		# define the [Fe/H] array, this is the values that the MIST 
		# and C3K grids are built
		self.FeHarr = ([-4.0,-3.5,-3.0,-2.75,-2.5,-2.25,-2.0,-1.75,
			-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5])

		# define the [alpha/Fe] array
		self.alphaarr = [0.0,0.2,0.4]

		# define aliases for the MIST isochrones and C3K/CKC files
		self.MISTpath = kwargs.get('MISTpath',Payne.__abspath__+'data/MIST/')
		self.C3Kpath  = kwargs.get('C3Kpath',Payne.__abspath__+'data/C3K/')

		# load MIST models
		self.MIST = h5py.File(self.MISTpath+'/MIST_1.2_EEPtrk.h5','r')
		self.MISTindex = list(self.MIST['index'])

		# create weights for Teff
		# determine the min/max Teff from MIST
		MISTTeffmin = np.inf
		MISTTeffmax = 0.0
		for ind in self.MISTindex:
			MISTTeffmin_i = self.MIST[ind]['log_Teff'].min()
			MISTTeffmax_i = self.MIST[ind]['log_Teff'].max()
			if MISTTeffmin_i < MISTTeffmin:
				MISTTeffmin = MISTTeffmin_i
			if MISTTeffmax_i > MISTTeffmax:
				MISTTeffmax = MISTTeffmax_i
		self.teffwgts = beta(0.5,0.5,loc=MISTTeffmin-10.0,scale=(MISTTeffmax+10.0)-(MISTTeffmin-10.0))

		# create weights for [Fe/H]
		self.fehwgts = beta(1.0,0.5,loc=-2.1,scale=2.7).pdf(self.FeHarr)
		self.fehwgts = self.fehwgts/np.sum(self.fehwgts)
			
		# create a dictionary for the C3K models and populate it for different
		# metallicities
		self.C3K = {}
		for aa in self.alphaarr:
			self.C3K[aa] = {}
			for mm in self.FeHarr:
				self.C3K[aa][mm] = h5py.File(
					self.C3Kpath+'/c3k_v1.3_feh{0:+4.2f}_afe{1:+3.1f}.full.h5'.format(mm,aa),
					'r')

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
		
		: params reclabelsel (optional):
			kwarg boolean that returns arrays that 
			give how the labels were selected

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
			Teffrange = [3000.0,15000.0]

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

		if 'reclabelsel' in kwargs:
			reclabelsel = kwargs['reclabelsel']
		else:
			reclabelsel = False

		if 'MISTweighting' in kwargs:
			MISTweighting = kwargs['MISTweighting']
		else:
			MISTweighting = False

		# randomly select num number of MIST isochrone grid points, currently only 
		# using dwarfs, subgiants, and giants (EEP = 202-605)

		labels = []
		spectra = []
		wavelength_o = []
		if reclabelsel:
			initlabels = []

		for ii in range(num):
			while True:
				# first randomly draw a [Fe/H]
				while True:
					if MISTweighting:
						p_i = self.fehwgts
					else:
						p_i = None
					FeH_i = np.random.choice(self.FeHarr,p=p_i)

					# check to make sure FeH_i is in user defined 
					# [Fe/H] limits
					if (FeH_i >= fehrange[0]) & (FeH_i <= fehrange[1]):
						break


				# then draw an alpha abundance
				while True:
					alpha_i = np.random.choice(self.alphaarr)

					# check to make sure alpha_i is in user defined
					# [alpha/Fe] limits
					if (alpha_i >= aFerange[0]) & (alpha_i <= aFerange[1]):
						break

				# select the C3K spectra at that [Fe/H] and [alpha/Fe]
				C3K_i = self.C3K[alpha_i][FeH_i]

				# create array of all labels in specific C3K file
				C3Kpars = np.array(C3K_i['parameters'])

				# select the range of MIST models with that [Fe/H]
				MIST_i = self.MIST['{0}/0.00/0.00'.format(FeH_i)]

				# restrict the MIST models to EEP = 202-606
				MIST_i = MIST_i[(MIST_i['EEP'] > 202) & (MIST_i['EEP'] < 606)]

				if MISTweighting:
					# generate Teff weights
					teffwgts_i = self.teffwgts.pdf(MIST_i['log_Teff'])
					teffwgts_i = teffwgts_i/np.sum(teffwgts_i)
				else:
					teffwgts_i = None


				while True:
					# randomly select a EEP, log(age) combination with weighting 
					# towards the hotter temps if user wants
					MISTsel = np.random.choice(len(MIST_i),p=teffwgts_i)

					# get MIST Teff and log(g) for this selection
					logt_MIST,logg_MIST = MIST_i['log_Teff'][MISTsel], MIST_i['log_g'][MISTsel]

					# check to make sure MIST log(g) and log(Teff) have a spectrum in the C3K grid
					# if not draw again
					if (
						(logt_MIST >= np.log10(Teffrange[0])) and (logt_MIST <= np.log10(Teffrange[1])) and
						(logg_MIST >= loggrange[0]) and (logg_MIST <= loggrange[1])
						):
						break

				# add a gaussian blur to the MIST selected Teff and log(g)
				# sigma_t = 750K, sigma_g = 1.5
				logt_MIST = np.log10(10.0**logt_MIST + np.random.randn()*750.0)
				logg_MIST = logg_MIST + np.random.randn()*1.5

				# do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
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
						outwave=wavelength_o,smoothtype='R',fftsmooth=True,inres=500000.0)
				else:
					spectra_i = spectra_i[wavecond]

				# check to see if labels are already in training set, if not store labels/spectrum
				if (label_i not in labels) and (not np.any(np.isnan(spectra_i))):
					labels.append(label_i)
					spectra.append(spectra_i)
					# if requested, record random selected parameters
					if reclabelsel:
						initlabels.append([logt_MIST,logg_MIST,FeH_i,alpha_i])
					break

		if reclabelsel:
			return np.array(spectra), np.array(labels), np.array(initlabels), wavelength_o
		else:
			return np.array(spectra), np.array(labels), wavelength_o

	def selspectra(self,inlabels,**kwargs):
		'''
		specifically select and return C3K spectra at user
		defined labels

		:param inlabels
		Array of user defined lables for returned C3K spectra
		format is [Teff,logg,FeH,aFe]

		'''

		if 'resolution' in kwargs:
			resolution = kwargs['resolution']
		else:
			resolution = None

		if 'waverange' in kwargs:
			waverange = kwargs['waverange']
		else:
			# default is just the MgB triplet 
			waverange = [5150.0,5200.0]

		labels = []
		spectra = []
		wavelength_o = []

		for li in inlabels:
			# select the C3K spectra at that [Fe/H] and [alpha/Fe]
			teff_i  = li[0]
			logg_i  = li[1]
			FeH_i   = li[2]
			alpha_i = li[3]

			C3K_i = self.C3K[alpha_i][FeH_i]

			# create array of all labels in specific C3K file
			C3Kpars = np.array(C3K_i['parameters'])

			# do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
			C3KNN = NearestNDInterpolator(
				np.array([C3Kpars['logt'],C3Kpars['logg']]).T,range(0,len(C3Kpars))
				)((teff_i,logg_i))

			# determine the labels for the selected C3K spectrum
			label_i = list(C3Kpars[C3KNN])

			# calculate the normalized spectrum
			spectra_i = C3K_i['spectra'][C3KNN]/C3K_i['continuua'][C3KNN]

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
					outwave=wavelength_o,smoothtype='R',fftsmooth=True,inres=500000.0)
			else:
				spectra_i = spectra_i[wavecond]

			labels.append(label_i)
			spectra.append(spectra_i)

		return np.array(spectra), np.array(labels), wavelength_o

	def smoothspecfunc(self,wave, spec, sigma, outwave=None, **kwargs):
		outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
		return outspec
