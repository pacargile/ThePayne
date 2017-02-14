# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import h5py
from multiprocessing import Pool
from scipy.interpolate import NearestNDInterpolator
import os,sys
from itertools import imap

# Theano is a very powerful package to train neural nets
# it performs "auto diff", i.e., provides analytic differentiation 
# of any cost function

import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid

from datetime import datetime


class TrainSpec(object):
	"""docstring for TrainSpec"""
	def __init__(self, **kwargs):

		# how many portions will we split the training data to 
		# perform stochastic gradient descent. Smaller batch 
		# size (i.e., larger number ) will take longer to converge.
		if 'mini_batch_size' in kwargs:
			self.mini_batch_size = kwargs['mini_batch_size']
		else:
			self.mini_batch_size = 2

		# initial step size in stochastic gradient descent.
		# smaller step size will be slower but will provide
		# better convergence.
		if 'eta_choice' in kwargs:
			self.eta_choice = kwargs['eta_choice']
		else:
			self.eta_choice = 0.1

		# the minimum step size beyond which we will truncate
		if 'min_eta' in kwargs:
			self.min_eta = kwargs['min_eta']
		else:
			self.min_eta = 0.001

		# how many steps of gradient descent per loop are going
		# to be performed
		if 'num_epochs_choice' in kwargs:
			self.num_epochs_choice = kwargs['num_epochs_choice']
		else:
			self.num_epochs_choice = 10000

		# truncation criteria
		if 'trunc_diff' in kwargs:
			self.trunc_diff = kwargs['trunc_diff']
		else:
			self.trunc_diff = 0.003

		# maximum number of loops (to aviod infinite loops)
		# i.e., max_iter*num_epoch_choice is the maximum number of steps 
		# beyond which we will truncate
		if 'max_iter' in kwargs:
			self.max_iter = kwargs['max_iter']
		else:
			self.max_iter = 100

		# how many neurons per layer
		# here we always consider two fully connected layers
		if 'n_neurons' in kwargs:
			self.n_neurons = kwargs['n_neurons']
		else:
			self.n_neurons = 10

		# number of training spectra
		if 'num_train' in kwargs:
			self.num_train = kwargs['num_train']
		else:
			self.num_train = 150

		# wavelength range to train on in angstroms
		if 'waverange' in kwargs:
			self.waverange = kwargs['waverange']
		else:
			self.waverange = [5150.0,5200.0]

		# check for user defined ranges for C3K spectra
		if 'Teff' in kwargs:
			Teffrange = kwargs['Teff']
		if 'logg' in kwargs:
			loggrange = kwargs['logg']
		if 'FeH' in kwargs:
			FeHrange = kwargs['FeH']

		# output hdf5 file name
		if 'output' in kwargs:
			self.outfilename = kwargs['output']
		else:
			self.outfilename = 'TESTOUT.h5'

		# pull C3K spectra for training
		self.spectra_o,self.labels_o,self.wavelength = self.pullspectra(self.num_train)

		self.labels_o = self.labels_o.T

		# trim spectra using the user defined wavelength range
		wavecond = (self.wavelength >= self.waverange[0]) & (self.wavelength <= self.waverange[1])
		wavecond = np.array(wavecond,dtype=bool)
		self.spectra_o = self.spectra_o[:,wavecond]
		self.wavelength = self.wavelength[wavecond]

		# neural-nets typically train a function mapping from 
		# [0,1] -> [0,1], so here we scale both input (labels) 
		# and output (fluxes) to [0.1,0.9]

		# record the min,max for the labels so that we can 
		# scale any labels to our training set
		self.x_min = np.min(self.labels_o,axis=1)
		self.x_max = np.max(self.labels_o,axis=1)

		# scale labels
		self.labels = ( (((self.labels_o.T - self.x_min)*0.8) / (self.x_max-self.x_min)) + 0.1).T

		# scale the fluxes, we assume model fluxes are already normalized
		self.spectra = self.spectra_o.T*0.8 + 0.1

		# initialize the output HDf5 file
		self.initout()

	def __call__(self,pixel_no):
		'''
		call instance so that train_pixel can be called with multiprocessing
		and still have all of the class instance variables
	
		:params pixel_no:
			Pixel number that is going to be trained

		'''
		return self.train_pixel(pixel_no)

	def run(self,mp=False,ncpus=1):
		'''
		function to actually run the training on pixels

		:param mp (optional):
			boolean that turns on multiprocessing to run 
			training in parallel

		'''
		# number of pixels to train
		numtrainedpixles = self.spectra.shape[0]

		# turn on multiprocessing if desired
		if mp:
			##### multiprocessing stuff #######
			numcpus = open('/proc/cpuinfo').read().count('processor\t:')
			os.system("taskset -p -c 0-{NCPUS} {PID}".format(NCPUS=numcpus-1,PID=os.getpid()))

			pool = Pool(processes=ncpus)
			# map out the pixel training
			# pool.map(self,range(numtrainedpixles))
			netout = pool.imap(self,range(numtrainedpixles))
			for ii,net in enumerate(netout):
				print(ii)
				sys.stdout.flush()
				# store and flush the network parameters into the HDF5 file
				self.w0_h5[ii,...] = net.layers[0].w.get_value().T
				self.b0_h5[ii,...] = net.layers[0].b.get_value()
				self.w1_h5[ii,...] = net.layers[1].w.get_value()[:,0]
				self.b1_h5[ii,...] = net.layers[1].b.get_value()[0]

				# flush the HDF5 file to store the output
				self.outfile.flush()

		else:
			# map out the pixel training
			netout = imap(self,range(numtrainedpixles))
			for ii,net in enumerate(netout):
				print(ii)
				sys.stdout.flush()
				# store and flush the network parameters into the HDF5 file
				self.w0_h5[ii,...] = net.layers[0].w.get_value().T
				self.b0_h5[ii,...] = net.layers[0].b.get_value()
				self.w1_h5[ii,...] = net.layers[1].w.get_value()[:,0]
				self.b1_h5[ii,...] = net.layers[1].b.get_value()[0]

				# flush the HDF5 file to store the output
				self.outfile.flush()

		# start total timer
		tottimestart = datetime.now()

		# print out total time
		print('Total time to train network: {0}'.format(datetime.now()-tottimestart))
		sys.stdout.flush()

		# # extract neural-net parameters
		# # the first layer
		# self.w_array_0 = np.array(
		# 	[net_array[i].layers[0].w.get_value().T
		# 	for i in range(numtrainedpixles)]
		# 	)
		# self.b_array_0 = np.array(
		# 	[net_array[i].layers[0].b.get_value()
		# 	for i in range(numtrainedpixles)]
		# 	)
		# # the second layer
		# self.w_array_1 = np.array(
		# 	[net_array[i].layers[1].w.get_value()[:,0]
		# 	for i in range(numtrainedpixles)]
		# 	)
		# self.b_array_1 = np.array(
		# 	[net_array[i].layers[1].b.get_value()[0]
		# 	for i in range(numtrainedpixles)]
		# 	)

		# formally close the output file
		self.outfile.close()

	def initout(self):
		'''
		function to save all of the information into 
		a single HDF5 file
		'''

		# create output HDF5 file
		self.outfile = h5py.File(self.outfilename,'w')

		# add datesets for values that are already defined
		self.wave_h5  = self.outfile.create_dataset('wavelength',data=self.wavelength,compression='gzip')
		self.label_h5 = self.outfile.create_dataset('labels',    data=self.labels_o,  compression='gzip')
		self.xmin_h5  = self.outfile.create_dataset('x_min',     data=self.x_min,     compression='gzip')
		self.xmax_h5  = self.outfile.create_dataset('x_max',     data=self.x_max,     compression='gzip')

		# create vectorized datasets for the netweork results to be added
		self.w0_h5    = self.outfile.create_dataset('w_array_0', (len(self.wavelength),10,3), compression='gzip')
		self.w1_h5    = self.outfile.create_dataset('w_array_1', (len(self.wavelength),10),   compression='gzip')
		self.b0_h5    = self.outfile.create_dataset('b_array_0', (len(self.wavelength),10),   compression='gzip')
		self.b1_h5    = self.outfile.create_dataset('b_array_1', (len(self.wavelength),),     compression='gzip')

		self.outfile.flush()


	def pullspectra(self, num, **kwargs):
		'''
		Randomly draw 2*num spectra from C3K based on 
		the MIST isochrones.
		
		:params num:
			Number of spectra randomly drawn for training

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

		# define the [Fe/H] array, this is the values that the MIST 
		# and C3K grids are built
		FeHarr = [-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5]

		# define aliases for the MIST isochrones and C3K/CKC files
		MISTpath = '/n/regal/conroy_lab/pac/ThePayne/models/MIST/'
		C3Kpath  = '/n/regal/conroy_lab/pac/ThePayne/models/CKC/'

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

		for ii in range(num):
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
					wavelength = np.array(C3K_i['wavelengths'])

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

				# check to see if labels are already in training set, if not store labels/spectrum
				if (label_i not in labels) and (not np.any(np.isnan(spectra_i))):
					labels.append(label_i)
					spectra.append(spectra_i)
					break

		return np.array(spectra), np.array(labels), wavelength


	def train_pixel(self,pixel_no):
		'''
		define training function for each wavelength pixel to run in parallel
		note we create individual neural network for each pixel
		'''

		# start a timer
		starttime = datetime.now()

		# extract flux of a wavelength pixel
		training_y = theano.shared(np.asarray(np.array([self.spectra[pixel_no,:]]).T, 
			dtype=theano.config.floatX))

		# convert labels into a theano variable
		training_x = theano.shared(np.asarray(self.labels.T,dtype=theano.config.floatX))

		# define the network
		net = Network([
			FullyConnectedLayer(
				n_in=training_x.get_value().shape[1],
				n_out=self.n_neurons),
			FullyConnectedLayer(
				n_in=self.n_neurons,
				n_out=training_y.get_value().shape[1]),
			], self.mini_batch_size)

		# initiate loop counter and step size
		loop_count = 0
		step_divide = 1.0

		# sometimes the network can get stuck at the initial point
		# so we first train for 1000 steps
		net.SGD(training_x,training_y,1000,self.eta_choice)

		# we evaluate if the cost has improved
		while (
			np.abs(np.mean(net.cost_train[:100])
			-np.mean(net.cost_train[-100:]))/(np.mean(net.cost_train[:100])) 
			< 0.1) and (loop_count < self.max_iter):

			# if not we reset the network (and hence the initial point)
			# and loop until it finds a valid initial point
			net = Network([
				FullyConnectedLayer(
					n_in=training_x.get_value().shape[1],
					n_out=self.n_neurons),
				FullyConnectedLayer(
					n_in=self.n_neurons,
					n_out=training_y.get_value().shape[1]),
				], self.mini_batch_size)
			net.SGD(training_x,training_y,1000,self.eta_choice)

			# increase counter
			loop_count += 1

		# after a good initial point is found, we proceed to the extensive training

		# initiate the deviation trunction criterion
		med_deviate = 1000.0

		# loop until the deviation is smaller than the chosen trunction criterion
		# we also truncate if the step size has become too small
		while (
			(med_deviate > self.trunc_diff) and (loop_count < self.max_iter) and (self.eta_choice/step_divide > self.min_eta)
			):

			# continue to train the network if it has not converged yet
			net.SGD(training_x,training_y,self.num_epochs_choice,self.eta_choice/step_divide)

			# increase counter to avoid infinite loop
			loop_count += 1

			# check if the current stepsize is too large, i.e., cost does not change much
			if (
				np.abs(np.mean(net.cost_train[:100])-np.mean(net.cost_train[-100:])) / (np.mean(net.cost_train[:100]))
				 < 0.01
				 ):

				# if so, we make the step size smaller
				step_divide = step_divide*2.0

			# this is the validation step
			# calculate the deviation between the analytic approximation vs. the training models
			# in principle, we should consider validation models here
			w_array_0 = net.layers[0].w.get_value().T
			b_array_0 = net.layers[0].b.get_value()
			w_array_1 = net.layers[1].w.get_value()[:,0]
			b_array_1 = net.layers[1].b.get_value()[0]

			predict_flux = act_func(
				np.sum(w_array_1*(act_func(np.dot(w_array_0,self.labels).T + b_array_0)), axis=1)
				+ b_array_1)

			# remember to scale back the fluxes to the normal metric
			### here we choose the maximum absolute deviation to be the truncation criteria ###
			med_deviate = np.max(np.abs((predict_flux-self.spectra[pixel_no,:])/0.8))

		print('Trained pixel:{0}/{1} (wavelength: {2}), took: {3}'.format(
			pixel_no,len(self.spectra[:,0]),self.wavelength[pixel_no],datetime.now()-starttime))
		sys.stdout.flush()

		# # store and flush the network parameters into the HDF5 file
		# self.w0_h5[pixel_no,...] = net.layers[0].w.get_value().T
		# self.b0_h5[pixel_no,...] = net.layers[0].b.get_value()
		# self.w1_h5[pixel_no,...] = net.layers[1].w.get_value()[:,0]
		# self.b1_h5[pixel_no,...] = net.layers[1].b.get_value()[0]

		# # flush the HDF5 file to store the output
		# self.outfile.flush()

		return net

class Network(object):
	"""
	Main Network Class
	"""
	def __init__(self, layers, mini_batch_size):
		self.layers = layers
		self.mini_batch_size = mini_batch_size
		self.params = [param for layer in self.layers for param in layer.params]
		
		self.x = T.dmatrix('x')
		self.y = T.dmatrix('y')

		init_layer = self.layers[0]
		init_layer.set_inpt(self.x,self)
		for j in xrange(1,len(self.layers)):
			prev_layer, layer = self.layers[j-1],self.layers[j]
			layer.set_inpt(prev_layer.output,self)
		self.output = self.layers[-1].output

		# create a property to record the cost function at each training step
		self.cost_train = []

	def SGD(self,training_x,training_y,epochs,eta):
		'''
		Stochastic Gradient Descent
		'''

		# reset the cost for each training loop
		self.cost_train = []

		# compute mini batches
		num_sample = training_x.get_value().shape[0]
		num_training_batches = num_sample/self.mini_batch_size

		# define cost function, symbolic graident, and updates
		cost = self.layers[-1].cost(self)
		grads = T.grad(cost,self.params)
		updates = ([(param, param-eta/self.mini_batch_size*grad) 
			for param, grad in zip(self.params,grads)])
		i=T.lscalar()

		# randomize the training data for stochastic gradient descent
		ind = np.arange(num_sample)
		np.random.shuffle(ind)
		ind = theano.shared(ind)

		# define function to train a mini-batch
		train_mb = theano.function(
			[i],cost, updates=updates,
			givens={
				self.x:training_x[ind[i*self.mini_batch_size:(i+1)*self.mini_batch_size]],
				self.y:training_y[ind[i*self.mini_batch_size:(i+1)*self.mini_batch_size]]
				}
			)

		# the actual training
		for epoch in xrange(epochs):
			cost_train_ij = 0.0
			for mini_batch_index in xrange(num_training_batches):
				# sum up all cost for each mini batch
				cost_train_ij += train_mb(mini_batch_index)
			self.cost_train.append(cost_train_ij)

class FullyConnectedLayer(object):
	'''
	Class that defines fully connected layers
	Here we choose sigmiod function to be the activation function
	'''

	def __init__(self, n_in,n_out,activation_fn=sigmoid):
		self.n_in = n_in
		self.n_out = n_out
		self.activation_fn = activation_fn

		# initialize weights and biases of the neural net
		self.w = theano.shared(
			np.asarray(
				np.random.normal(
					loc=0.0,scale=np.sqrt(1.0/n_out),
					size=(n_in,n_out)
					),
				dtype=theano.config.floatX
				)
			)

		self.b = theano.shared(
			np.asarray(
				np.random.normal(
					loc=0.0,scale=1.0,
					size=(n_out,)
					),
				dtype=theano.config.floatX
				)
			)
		self.params = [self.w,self.b]

	def set_inpt(self,inpt,net):
		'''
		Define input and output for each neural net layer
		'''
		self.inpt = inpt.reshape((net.mini_batch_size,self.n_in))
		self.output = self.activation_fn(T.dot(self.inpt,self.w) + self.b)

	def cost(self,net):
		'''
		Define a cost function
		'''
		return T.sum(T.abs_(net.y-self.output))

def act_func(z):
	'''
	Define action function that we will use in the 
	validation step. Make sure this function is 
	consistent with the training function.

	:params z:
	label that goes into the sigmoid

	'''
	return 1.0/(1.0+np.exp(-z))
