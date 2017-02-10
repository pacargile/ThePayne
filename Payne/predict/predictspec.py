# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from scipy import constants
speedoflight = constants.c / 1000.0

from ..utils.smoothing import smoothspec


class PaynePredict(object):
	"""
	Class for taking a Payne-learned NN and predicting spectrum.
	"""
	def __init__(self, NNfilename):
		self.NN = {}
		# name of file that contains the neural-net output
		self.NN['filename'] = NNfilename
		# restrore hdf5 file with the NN
		self.NN['file']     = h5py.File(self.NN['filename'])
		# wavelength for predicted spectrum
		self.NN['wavelength']  = np.array(self.NN['file']['wavelength'])
		# labels for which the NN was trained on, useful to make
		# sure prediction is within the trained grid.
		self.NN['labels']   = np.array(self.NN['file']['labels'])
		# NN coefficents for action function
		self.NN['w0'] = np.array(self.NN['file']['w_array_0'])
		self.NN['w1'] = np.array(self.NN['file']['w_array_1'])
		self.NN['b0'] = np.array(self.NN['file']['b_array_0'])
		self.NN['b1'] = np.array(self.NN['file']['b_array_1'])
		# label bounds
		self.NN['xmin'] = np.array(self.NN['file']['x_min'])
		self.NN['xmax'] = np.array(self.NN['file']['x_max'])

	def act_func(self,z):
		'''
		Action function, default is a sigmoid. May want to 
		change this so that it checks to see if action 
		function is included in the NN file

		:params z:
		learned paramter used to define the NN sigmoids
		
		:returns sigmoid:
		NN sigmoids
		'''
		return 1.0/(1.0+np.exp(-z))

	def predictspec(self,labels):
		'''
		predict spectra using set of labels, NN output, and
		the action function

		:params labels:
		list of label values for the labels used to train the NN
		ex. [Teff,log(g),[Fe/H]]

		:returns predict_flux:
		predicted flux from the NN
		'''

		# assume that the labels are not scaled to the NN, must 
		# rescale them according to xmin, xmax

		# first check if labels are within the trained network, warn if outside
		if any(labels-self.NN['xmin'] < 0.0) or any(self.NN['xmax']-labels < 0.0):
			print 'WARNING: user labels are outside the trained network!!!'

		slabels = ((labels-self.NN['xmin'])*0.8)/(self.NN['xmax']-self.NN['xmin']) + 0.1

		predict_flux = self.act_func(
			np.sum(
				self.NN['w1']*(self.act_func(np.dot(self.NN['w0'],slabels) + self.NN['b0'])),
				axis=1)
			+ self.NN['b1'])

		# scale predicted flux back to normal
		predict_flux = (predict_flux-0.1)/0.8

		return predict_flux

	def getspec(self,**kwargs):
		'''
		function to take a set of kwarg based on labels and 
		return the predicted spectrum
		
		default returns solar spectrum, rotating at 2 km/s, and 
		at R=32K

		: returns modwave:
		Wavelength array from the NN

		:returns modspec:
		Predicted spectrum from the NN

		'''

		self.inputdict = {}

		if 'Teff' in kwargs:
			self.inputdict['logt'] = np.log10(kwargs['Teff'])
		elif 'logt' in kwargs:
			self.inputdict['logt'] = kwargs['logt']
		else:
			self.inputdict['logt'] = np.log10(5770.0)

		if 'log(g)' in kwargs:
			self.inputdict['logg'] = kwargs['log(g)']
		elif 'logg' in kwargs:
			self.inputdict['logg'] = kwargs['logg']
		else:
			self.inputdict['logg'] = 4.44
		
		if '[Fe/H]' in kwargs:
			self.inputdict['feh'] = kwargs['[Fe/H]']
		elif 'feh' in kwargs:
			self.inputdict['feh'] = kwargs['feh']
		else:
			self.inputdict['feh'] = 0.0
		
		# calculate model spectrum at the native C3K resolution
		modspec = self.predictspec([self.inputdict[kk] for kk in ['logt','logg','feh']])

		if 'outwave' in kwargs:
			# define array of output wavelength values
			outwave = kwargs['outwave']
		else:
			outwave = self.NN['wavelength']

		if 'rot_vel' in kwargs:
			# check to make sure rot_vel isn't 0.0, this will cause the convol. to crash
			if kwargs['rot_vel'] != 0.0:
				# use BJ's smoothspec to convolve with rotational broadening
				modspec = self.smoothspec(self.NN['wavelength'],modspec,kwargs['rot_vel'],
					outwave=outwave,smoothtype='vel',fftsmooth=True)

		if 'rad_vel' in kwargs:
			# kwargs['radial_velocity']: RV in km/s
			modwave = self.NN['wavelength'].copy()*(1.0-(kwargs['rad_vel']/speedoflight))
		else:
			modwave = self.NN['wavelength']

		if 'inst_R' in kwargs:
			# check to make sure inst_R != 0.0
			if kwargs['inst_R'] != 0.0:
				# instrumental broadening
				modspec = self.smoothspec(modwave,modspec,kwargs['inst_R'],
					outwave=outwave,smoothtype='R',fftsmooth=True)

		return modwave, modspec

	def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
		outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
		return outspec