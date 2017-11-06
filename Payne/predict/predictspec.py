# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import h5py
from scipy import constants
speedoflight = constants.c / 1000.0
from scipy.interpolate import UnivariateSpline

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
		self.NN['file']     = h5py.File(self.NN['filename'],'r')
		# wavelength for predicted spectrum
		self.NN['wavelength']  = np.array(self.NN['file']['wavelength'])
		# labels for which the NN was trained on, useful to make
		# sure prediction is within the trained grid.

		# check to see if any wavelengths are == 0.0
		goodruncond = self.NN['wavelength'] != 0.0
		self.NN['wavelength'] = self.NN['wavelength'][goodruncond]

		self.NN['labels']   = np.array(self.NN['file']['labels'])
		# resolution that network was trained at
		self.NN['resolution'] = np.array(self.NN['file']['resolution'])[0]
		# NN coefficents for action function
		self.NN['w0'] = np.array(self.NN['file']['w_array_0'])[goodruncond]
		self.NN['w1'] = np.array(self.NN['file']['w_array_1'])[goodruncond]
		self.NN['b0'] = np.array(self.NN['file']['b_array_0'])[goodruncond]
		self.NN['b1'] = np.array(self.NN['file']['b_array_1'])[goodruncond]
		# label bounds
		self.NN['x_min'] = np.array(self.NN['file']['x_min'])
		self.NN['x_max'] = np.array(self.NN['file']['x_max'])

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
		ex. [Teff,log(g),[Fe/H],[alpha/Fe]]

		:returns predict_flux:
		predicted flux from the NN
		'''

		# assume that the labels are not scaled to the NN, must 
		# rescale them according to xmin, xmax

		# first check if labels are within the trained network, warn if outside
		if any(labels-self.NN['x_min'] < 0.0) or any(self.NN['x_max']-labels < 0.0):
			print('WARNING: user labels are outside the trained network!!!')
			print('WARNING: Here are the problematic labels: ',labels)
			print('WARNING: Here are the NN[x_min]: ',self.NN['x_min'])
			print('WARNING: Here are the NN[x_max]: ',self.NN['x_max'])

		slabels = ((labels-self.NN['x_min'])*0.8)/(self.NN['x_max']-self.NN['x_min']) + 0.1

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

		if '[alpha/Fe]' in kwargs:
			self.inputdict['afe'] = kwargs['[alpha/Fe]']
		elif 'aFe' in kwargs:
			self.inputdict['afe'] = kwargs['aFe']
		elif 'afe' in kwargs:
			self.inputdict['afe'] = kwargs['afe']
		else:
			self.inputdict['afe'] = 0.0
		
		# calculate model spectrum at the native network resolution
		modspec = self.predictspec([self.inputdict[kk] for kk in ['logt','logg','feh','afe']])

		modwave = self.NN['wavelength']

		rot_vel_bool = False
		if 'rot_vel' in kwargs:
			# check to make sure rot_vel isn't 0.0, this will cause the convol. to crash
			if kwargs['rot_vel'] != 0.0:
				# set boolean to let rest of code know the spectrum has been broadened
				rot_vel_bool = True
				# use B.Johnson's smoothspec to convolve with rotational broadening
				modspec = self.smoothspec(modwave,modspec,kwargs['rot_vel'],
					outwave=None,smoothtype='vsini',fftsmooth=True)

				# TEST THE FOLLOWING
				# # apply macroturbulent broadening using Doyle et al. 2014 relationship
				# vmac = (
				# 	3.21 + 
				# 	2.33 * (10.0**−3.0)*(10.0**self.inputdict['logt'] − 5777.0) + 
				# 	2.00 * (10.0**−6.0)*(10.0**self.inputdict['logt'] − 5777.0)**2.0 - 
				# 	2.00 * (self.inputdict['logg'] − 4.44)
				# 	)
				# modspec = self.smoothspec(modwave,modspec,vmac,
				# 	outwave=None,smoothtype='vel',fftsmooth=True)

		rad_vel_bool = False
		if 'rad_vel' in kwargs:
			if kwargs['rad_vel'] != 0.0:
				# kwargs['radial_velocity']: RV in km/s
				rad_vel_bool = True
				# modwave = self.NN['wavelength'].copy()*(1.0-(kwargs['rad_vel']/speedoflight))
				modwave = modwave*(1.0+(kwargs['rad_vel']/speedoflight))

		inst_R_bool = False
		if 'inst_R' in kwargs:
			# check to make sure inst_R != 0.0
			if kwargs['inst_R'] != 0.0:
				inst_R_bool = True
				# instrumental broadening
				# if rot_vel_bool:
				# 	inres = (2.998e5)/kwargs['rot_vel']
				# else:
				# 	inres = self.NN['resolution']
				# inres=None
				if 'outwave' in kwargs:
					if type(kwargs['outwave']) == type(None):
						outwave = None
					else:
						outwave = np.array(kwargs['outwave'])
				else:
					outwave = None

				modspec = self.smoothspec(modwave,modspec,kwargs['inst_R'],
					outwave=outwave,smoothtype='R',fftsmooth=True,inres=self.NN['resolution'])
				if type(outwave) != type(None):
					modwave = outwave
		if (inst_R_bool == False) & ('outwave' in kwargs):
			modspec = UnivariateSpline(modwave,modspec,k=1,s=0)(kwargs['outwave'])
			modwave = kwargs['outwave']

		return modwave, modspec

	def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
		outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
		return outspec