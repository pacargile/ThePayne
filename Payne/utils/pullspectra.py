# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys,glob,warnings
import numpy as np
with warnings.catch_warnings():
	warnings.simplefilter('ignore')
	import h5py
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import beta
from datetime import datetime

import Payne
from .smoothing import smoothspec

class pullspectra(object):
	def __init__(self,**kwargs):
		# define aliases for the MIST isochrones and C3K/CKC files
		self.MISTpath = kwargs.get('MISTpath',Payne.__abspath__+'data/MIST/')
		self.C3Kpath  = kwargs.get('C3Kpath',Payne.__abspath__+'data/C3K/')

		if type(self.MISTpath) == type(None):
			self.MISTpath = Payne.__abspath__+'data/MIST/'

		if type(self.C3Kpath) == type(None):
			self.C3Kpath = Payne.__abspath__+'data/C3K/'

		# load MIST models
		self.MIST = h5py.File(self.MISTpath+'/MIST_1.2_EEPtrk.h5','r')
		self.MISTindex = list(self.MIST['index'])
		# convert btye strings to python strings
		self.MISTindex = [x.decode("utf-8") for x in self.MISTindex]

		self.FeHarr = []
		self.alphaarr = []

		# determine the FeH and aFe arrays for C3K
		for indinf in glob.glob(self.C3Kpath+'*'):
			self.FeHarr.append(float(indinf.split('_')[-2][3:]))
			self.alphaarr.append(float(indinf.split('_')[-1][3:-8]))

		# remove the super metal-rich models that only have aFe = 0
		self.FeHarr.remove(0.75)
		self.FeHarr.remove(1.00)
		self.FeHarr.remove(1.25)

		# determine the MIST FeH and aFe arrays
		self.MISTFeHarr = []
		self.MISTalphaarr = []
		for indinf in self.MISTindex:
			self.MISTFeHarr.append(float(indinf.split('/')[0]))
			self.MISTalphaarr.append(float(indinf.split('/')[1]))

		# create weights for Teff
		# determine the min/max Teff from MIST
		self.MISTTeffmin = np.inf
		self.MISTTeffmax = 0.0
		for ind in self.MISTindex:
			MISTTeffmin_i = self.MIST[ind]['log_Teff'].min()
			MISTTeffmax_i = self.MIST[ind]['log_Teff'].max()
			if MISTTeffmin_i < self.MISTTeffmin:
				self.MISTTeffmin = MISTTeffmin_i
			if MISTTeffmax_i > self.MISTTeffmax:
				self.MISTTeffmax = MISTTeffmax_i

		self.teffwgts = {}
		for ind in self.MISTindex:
			self.teffwgts[ind] = beta(0.2,1.5,
				loc=self.MISTTeffmin-0.1,
				scale=(self.MISTTeffmax+0.1)-(self.MISTTeffmin-0.1)
				).pdf(self.MIST[ind]['log_Teff'])
			self.teffwgts[ind] = self.teffwgts[ind]/np.sum(self.teffwgts[ind])
		# self.teffwgts = beta(0.5,1.0,loc=self.MISTTeffmin-0.1,scale=(self.MISTTeffmax+0.1)-(self.MISTTeffmin-0.1))

		# create weights for [Fe/H]
		self.fehwgts = beta(1.0,0.2,loc=-4.1,scale=4.7).pdf(self.FeHarr)
		self.fehwgts = self.fehwgts/np.sum(self.fehwgts)
			
		# create a dictionary for the C3K models and populate it for different
		# metallicities
		self.C3K = {}
		for aa in self.alphaarr:
			self.C3K[aa] = {}
			for mm in self.FeHarr:
				self.C3K[aa][mm] = h5py.File(
					self.C3Kpath+'c3k_v1.3_feh{0:+4.2f}_afe{1:+3.1f}.full.h5'.format(mm,aa),
					'r', libver='latest', swmr=True)

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
			Teffrange = [2500.0,15000.0]

		if 'logg' in kwargs:
			loggrange = kwargs['logg']
		else:
			loggrange = [-1.0,5.0]

		if 'FeH' in kwargs:
			fehrange = kwargs['FeH']
		else:
			fehrange = [min(self.FeHarr),max(self.FeHarr)]

		if 'aFe' in kwargs:
			aFerange = kwargs['aFe']
		else:
			aFerange = [min(self.alphaarr),max(self.alphaarr)]

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

		if 'timeit' in kwargs:
			timeit = kwargs['timeit']
		else:
			timeit = False

		# randomly select num number of MIST isochrone grid points, currently only 
		# using dwarfs, subgiants, and giants (EEP = 202-605)

		labels = []
		spectra = []
		wavelength_o = []
		if reclabelsel:
			initlabels = []


		for ii in range(num):
			if timeit:
				starttime = datetime.now()

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
				if timeit:
					print('Pulled random [Fe/H] & [a/Fe] in {0}'.format(datetime.now()-starttime))

				# select the C3K spectra at that [Fe/H] and [alpha/Fe]
				C3K_i = self.C3K[alpha_i][FeH_i]

				# create array of all labels in specific C3K file
				C3Kpars = np.array(C3K_i['parameters'])

				if timeit:
					print('create arrray of C3Kpars in {0}'.format(datetime.now()-starttime))

				# select the range of MIST models with that [Fe/H]
				# first determine the FeH and aFe that are nearest to MIST values
				FeH_i_MIST = self.MISTFeHarr[np.argmin(np.abs(FeH_i-self.MISTFeHarr))]
				aFe_i_MIST = self.MISTalphaarr[np.argmin(np.abs(alpha_i-self.MISTalphaarr))]
				MIST_i = self.MIST['{0:4.2f}/{1:4.2f}/0.00'.format(FeH_i_MIST,aFe_i_MIST)]

				if timeit:
					print('Pulled MIST models in {0}'.format(datetime.now()-starttime))

				if MISTweighting:
					# # generate Teff weights
					# teffwgts_i = self.teffwgts.pdf(MIST_i['log_Teff'])
					# teffwgts_i = teffwgts_i/np.sum(teffwgts_i)
					teffwgts_i = self.teffwgts['{0:4.2f}/{1:4.2f}/0.00'.format(FeH_i_MIST,aFe_i_MIST)]
				else:
					teffwgts_i = None

				if timeit:
					print('Created MIST weighting {0}'.format(datetime.now()-starttime))

				while True:
					# randomly select a EEP, log(age) combination with weighting 
					# towards the hotter temps if user wants
					MISTsel = np.random.choice(len(MIST_i),p=teffwgts_i)

					# get MIST Teff and log(g) for this selection
					logt_MIST_i,logg_MIST_i = MIST_i[MISTsel]['log_Teff'], MIST_i[MISTsel]['log_g']

					# check to make sure MIST log(g) and log(Teff) have a spectrum in the C3K grid
					# if not draw again
					if (
						(logt_MIST_i >= np.log10(Teffrange[0])) and (logt_MIST_i <= np.log10(Teffrange[1])) and
						(logg_MIST_i >= loggrange[0]) and (logg_MIST_i <= loggrange[1])
						):
						break
				if timeit:
					print('Selected MIST pars in {0}'.format(datetime.now()-starttime))


				# add a gaussian blur to the MIST selected Teff and log(g)
				# sigma_t = 750K, sigma_g = 1.5
				randomT = np.random.randn()*1000.0
				randomg = np.random.randn()*1.5

				# check to see if randomT is an issue for log10
				if 10.0**logt_MIST_i + randomT <= 0.0:
					randomT = np.abs(randomT)
					
				with warnings.catch_warnings():
					warnings.filterwarnings('error')
					try:
						logt_MIST = np.log10(10.0**logt_MIST_i + randomT)				
						logg_MIST = logg_MIST_i + randomg
					except Warning:
						print(
							'Caught a MIST parameter that does not make sense: {0} {1} {2} {3}'.format(
								randomT,10.0**logt_MIST_i,randomg,logg_MIST_i))
						logt_MIST = logt_MIST_i
						logg_MIST = logg_MIST_i

				# do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
				C3KNN = NearestNDInterpolator(
					np.array([C3Kpars['logt'],C3Kpars['logg']]).T,range(0,len(C3Kpars))
					)((logt_MIST,logg_MIST))

				# determine the labels for the selected C3K spectrum
				label_i = list(C3Kpars[C3KNN])

				if timeit:
					print('Determine C3K labels in {0}'.format(datetime.now()-starttime))

				# check to see if user defined labels to exclude, if so
				# continue on to the next iteration
				if list(label_i) in excludelabels:
					continue

				# turn off warnings for this step, C3K has some continuaa with flux = 0
				with np.errstate(divide='ignore', invalid='ignore'):
					spectra_i = C3K_i['spectra'][C3KNN]/C3K_i['continuua'][C3KNN]

				if timeit:
					print('Create C3K spectra in {0}'.format(datetime.now()-starttime))

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

				if timeit:
					print('Saved a C3K wavelength instance in {0}'.format(datetime.now()-starttime))

				# if user defined resolution to train at, the smooth C3K to that resolution
				if resolution != None:
					spectra_i = self.smoothspecfunc(wavelength_i,spectra_i,resolution,
						outwave=wavelength_o,smoothtype='R',fftsmooth=True,inres=500000.0)
				else:
					spectra_i = spectra_i[wavecond]

				if timeit:
					print('Convolve C3K to new R in {0}'.format(datetime.now()-starttime))

				# check to see if labels are already in training set, if not store labels/spectrum
				if (label_i not in labels) and (not np.any(np.isnan(spectra_i))):
					labels.append(label_i)
					spectra.append(spectra_i)
					# if requested, record random selected parameters
					if reclabelsel:
						initlabels.append([logt_MIST,logg_MIST,FeH_i,alpha_i])
					break
			if timeit:
				print('TOTAL TIME: {0}'.format(datetime.now()-starttime))
				print('')
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

		if isinstance(inlabels[0],float):
			inlabels = [inlabels]

		for li in inlabels:
			# select the C3K spectra at that [Fe/H] and [alpha/Fe]
			teff_i  = li[0]
			logg_i  = li[1]
			FeH_i   = li[2]
			alpha_i = li[3]

			# find nearest value to FeH and aFe
			FeH_i   = self.FeHarr[np.argmin(np.abs(np.array(self.FeHarr)-FeH_i))]
			alpha_i = self.alphaarr[np.argmin(np.abs(np.array(self.alphaarr)-alpha_i))]

			# select the C3K spectra for these alpha and FeH
			C3K_i = self.C3K[alpha_i][FeH_i]

			# create array of all labels in specific C3K file
			C3Kpars = np.array(C3K_i['parameters'])

			# do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
			C3KNN = NearestNDInterpolator(
				np.array([C3Kpars['logt'],C3Kpars['logg']]).T,range(0,len(C3Kpars))
				)((teff_i,logg_i))

			# determine the labels for the selected C3K spectrum
			label_i = list(C3Kpars[C3KNN])

			# turn off warnings for this step, C3K has some continuaa with flux = 0
			with np.errstate(divide='ignore', invalid='ignore'):
				spectra_i = C3K_i['spectra'][C3KNN]/C3K_i['continuua'][C3KNN]

			# check to see if label_i in labels, or spectra_i is nan's
			# if so, then skip the append and go to next step in while loop
			# do this before the smoothing to reduce run time
			# if np.any(np.isnan(spectra_i)):
			# 	continue

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

	def pullpixel(self,pixelnum,**kwargs):
		# a convience function if you only want to pull one pixel at a time
		# it also does a check and remove any NaNs in spectra 

		if 'Teff' in kwargs:
			Teffrange = kwargs['Teff']
		else:
			Teffrange = [2500.0,15000.0]

		if 'logg' in kwargs:
			loggrange = kwargs['logg']
		else:
			loggrange = [-1.0,5.0]

		if 'FeH' in kwargs:
			fehrange = kwargs['FeH']
		else:
			fehrange = [min(self.FeHarr),max(self.FeHarr)]

		if 'aFe' in kwargs:
			aFerange = kwargs['aFe']
		else:
			aFerange = [min(self.alphaarr),max(self.alphaarr)]

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
			waverange = [5145.0,5300.0]

		if 'reclabelsel' in kwargs:
			reclabelsel = kwargs['reclabelsel']
		else:
			reclabelsel = False

		if 'MISTweighting' in kwargs:
			MISTweighting = kwargs['MISTweighting']
		else:
			MISTweighting = True

		if 'timeit' in kwargs:
			timeit = kwargs['timeit']
		else:
			timeit = False

		if 'inlabels' in kwargs:
			inlabels = kwargs['inlabels']
		else:
			inlabels = []

		if 'num' in kwargs:
			num = kwargs['num']
		else:
			num = 1

		if inlabels == []:				
			# pull the spectrum
			spectra,labels,wavelength = self(
				num,resolution=resolution, waverange=waverange,
				MISTweighting=MISTweighting)

		else:
			spectra,labels,wavelength = self.selspectra(
				inlabels,resolution=resolution, waverange=waverange)

		# select individual pixels
		pixelarr = np.array(spectra[:,pixelnum])
		labels = np.array(labels)

		# # determine if an of the pixels are NaNs
		# mask = np.ones_like(pixelarr,dtype=bool)
		# nanval = np.nonzero(np.isnan(pixelarr))
		# numnan = len(nanval[0])
		# mask[np.nonzero(np.isnan(pixelarr))] = False

		# # remove nan pixel values and labels
		# pixelarr = pixelarr[mask]
		# labels = labels[mask]
		
		return pixelarr, labels, wavelength

	def checklabels(self,inlabels,**kwargs):
		# a function that allows the user to determine the nearest C3K labels to array on input labels
		# useful to run before actually selecting spectra

		labels = []

		for li in inlabels:
			# select the C3K spectra at that [Fe/H] and [alpha/Fe]
			teff_i  = li[0]
			logg_i  = li[1]
			FeH_i   = li[2]
			alpha_i = li[3]

			# find nearest value to FeH and aFe
			FeH_i   = self.FeHarr[np.argmin(np.abs(self.FeHarr-FeH_i))]
			alpha_i = self.alphaarr[np.argmin(np.abs(self.alphaarr-alpha_i))]

			# select the C3K spectra for these alpha and FeH
			C3K_i = self.C3K[alpha_i][FeH_i]

			# create array of all labels in specific C3K file
			C3Kpars = np.array(C3K_i['parameters'])

			# do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
			C3KNN = NearestNDInterpolator(
				np.array([C3Kpars['logt'],C3Kpars['logg']]).T,range(0,len(C3Kpars))
				)((teff_i,logg_i))

			# determine the labels for the selected C3K spectrum
			label_i = list(C3Kpars[C3KNN])		
			labels.append(label_i)

		return np.array(labels)


	def smoothspecfunc(self,wave, spec, sigma, outwave=None, **kwargs):
		outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
		return outspec
