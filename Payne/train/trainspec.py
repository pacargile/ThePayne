# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from multiprocessing import Pool

import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid

class TrainSpec(object):
	"""docstring for TrainSpec"""
	def __init__(self, **kwargs):

		# how many portions will we splot the training data to 
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
			self.num_train = 200

		# pull C3K spectra for training
		self.spectra,self.labels = pullspectra(self.num_train)

		# neural-nets typically train a function mapping from 
		# [0,1] -> [0,1], so here we scale both input (labels) 
		# and output (fluxes) to [0.1,0.9]

		# record the min,max for the labels so that we can 
		# scale any labels to our training set
		self.x_min = np.min(self.labels,axis=1)
		self.x_max = np.max(self.labels,axis=1)

		# scale labels
		self.labels = ( (((self.labels.T - self.x_min)*0.8) / (self.x_max-self.x_min)) + 0.1).T

		# scale the fluxes, we assume model fluxes are already normalized
		self.spectra = self.spectra.T*0.8 + 0.1

		# Theano is a very powerful package to train neural nets
		# it performs "auto diff", i.e., provides analytic differentiation 
		# of any cost function

		# convert labels into a theano variable
		self.training_x = theano.shared(np.asarray(self.labels.T,dtype='float64'))


	def pullspectra(num, **kwargs):
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

		### LOAD C3K SPECTRA ####
		### LOAD MIST MODELS ####
		### RANDOMLY SELECT STARS FROM MIST ###
		### DO NEAREST NEIGHBOR INTERPOLATION ON C3K TO FIND SPECTRA & LABELS ###

		spectra = None # spectra -> structured array: wave, spectra1, spectra2, spectra3, ...
		labels = None # labels -> array of [labels1,labels2,labels3,...]

		return spectra, labels


	def act_func(z):
		'''
		Define action function that we will use in the 
		validation step. Make sure this function is 
		consistent with the training function.
		
		:params z:
		label that goes into the sigmoid
	
		'''
		return 1.0/(1.0+np.exp(-z))

class Network(object):
	"""
	Main Network Class
	"""
	def __init__(self, layers, mini_batch_size):
		self.arg = layers
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
		train_mb - theano.function(
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

def train_pixel(pixel_no):
	'''
	define training function for each wavelength pixel to run in parallel
	note we create individual neural network for each pixel
	'''

	# extract flux of a wavelength pixel
	training_y = theano.shared(np.asarray(np.array([spectra[pixel_no,:]]).T, dtype='float64'))

	# define the network
	net = Network([
		FullyConnectedLayer(
			n_in=training_x.get_value().shape[1],
			n_out=n_neurons),
		FullyConnectedLayer(
			n_in=n_neurons,
			n_out=training_y.get_value().shape[1]),
		], mini_batch_size)

	# initiate loop counter and step size
	loop_count = 0
	step_divide = 1.0

	# sometimes the network can get stuck at the initial point
	# so we first train for 1000 steps
	net.SGD(training_x,training_y,1000,eta_choice)

	# we evaluate if the cost has improved
	while (
		np.abs(np.mean(net.cost_train[:100])
		-np.mean(net.cost_train[-100:]))/(np.mean(net.cost_train[:100])) 
		< 0.1) and (loop_count < max_iter):

		# if not we reset the network (and hence the initial point)
		# and loop until it finds a valid initial point
		net = Network([
			FullyConnectedLayer(
				n_in=training_x.get_value().shape[1],
				n_out=n_neurons),
			FullyConnectedLayer(
				n_in=n_neurons,
				n_out=training_y.get_value().shape[1]),
			], mini_batch_size)
		net.SGD(training_x,training_y,1000,eta_choice)

		# increase counter
		loop_count += 1

	# after a good initial point is found, we proceed to the extensive training

	# initiate the deviation trunction criterion
	med_deviate = 1000.0

	# loop until the deviation is smaller than the chosen trunction criterion
	# we also truncate if the step size has become too small
	while (
		(med_deviate > trunc_diff) and (loop_count < max_iter) and (eta_choice/step_divide > min_eta)
		):

		# continue to train the network if it has not converged yet
		net.SGD(training_x,training_y,num_epochs_choice,eta_choice/step_divide)

		# increase counter to avoid infinite loop
		loop_count += 1

		# check if the current stepsize is too large, i.e., cost does not change much
		if (
			np.abs(np.mean(net.cost_train[:100])-np.mean(np.cost_train[-100:])) / (np.mean(net.cost_train[:100]))
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
			np.sum(
				w_array_1*act_func(np.dot(w_array_0,labels).T + b_array_0)
				,axis=1)
			+ b_array_1)

		# remember to scale back the fluxes to the normal metric
		### here we choose the maximum absolute deviation to be the truncation criteria ###
		med_deviate = np.max(np.abs((predict_flux-spectra[pixel_no,:])/0.8))

	# return the trained network for this pixel
	return net