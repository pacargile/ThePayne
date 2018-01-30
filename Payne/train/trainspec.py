import torch
from torch import nn
dtype = torch.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np
import h5py
import time,sys,os
from datetime import datetime
from itertools import imap
from multiprocessing import Pool

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
matplotlib.pyplot.ioff()

from ..utils.pullspectra import pullspectra

class Net(nn.Module):  
	def __init__(self, D_in, H, D_out):
		super(Net, self).__init__()
		self.lin1 = nn.Linear(D_in, H)
		self.lin2 = nn.Linear(H,H)
		self.lin3 = nn.Linear(H, D_out)

	def forward(self, x):
		x_i = self.encode(x)
		out1 = F.sigmoid(self.lin1(x_i))
		out2 = F.sigmoid(self.lin2(out1))
		y_i = self.lin3(out2)
		return y_i     

	def encode(self,x):
		# convert x into numpy to do math
		x_np = x.data.numpy()
		try:
			self.xmin
			self.xmax
		except (NameError,AttributeError):
			self.xmin = np.amin(x_np,axis=0)
			self.xmax = np.amax(x_np,axis=0)

		x = (x_np-self.xmin)/(self.xmax-self.xmin)
		return Variable(torch.from_numpy(x).type(dtype))

class TrainSpec(object):
	"""docstring for TrainSpec"""
	def __init__(self, **kwargs):
		# number of models to train on
		if 'numtrain' in kwargs:
			self.numtrain = kwargs['numtrain']
		else:
			self.numtrain = 20000

		# number of iteration steps in training
		if 'niter' in kwargs:
			self.niter = kwargs['niter']
		else:
			self.niter = 200000

		if 'epochs' in kwargs:
			self.epochs = kwargs['epochs']
		else:
			self.epochs = 5

		# number of nuerons in each layer
		if 'H' in kwargs:
			self.H = kwargs['H']
		else:
			self.H = 256

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
		if 'aFe' in kwargs:
			aFerange = kwargs['aFe']

		if 'resolution' in kwargs:
			self.resolution = kwargs['resolution']
		else:
			self.resolution = None

		if 'verbose' in kwargs:
			self.verbose = kwargs['verbose']
		else:
			self.verbose = False

		# output hdf5 file name
		if 'output' in kwargs:
			self.outfilename = kwargs['output']
		else:
			self.outfilename = 'TESTOUT.h5'

		if 'restartfile' in kwargs:
			self.restartfile = kwargs['restartfile']
			print('... Using Restart File: {0}'.format(self.restartfile))
		else:
			self.restartfile = None

		if 'saveopt' in kwargs:
			self.saveopt = kwargs['saveopt']
		else:
			self.saveopt = False

		if 'logepoch' in kwargs:
			self.logepoch = kwargs['logepoch']
		else:
			self.logepoch = True

		if 'adaptivetrain' in kwargs:
			self.adaptivetrain = kwargs['adaptivetrain']
		else:
			self.adaptivetrain = True

		# pull C3K spectra for training
		print('... Pulling Training Spectra')
		sys.stdout.flush()
		self.pullspectra = pullspectra()
		self.spectra_o,self.labels_o,self.wavelength = self.pullspectra(
			self.numtrain,resolution=self.resolution, waverange=self.waverange,
			MISTweighting=True)

		self.spectra = self.spectra_o.T
		  
		# N is batch size (number of points in X_train),
		# D_in is input dimension
		# H is hidden dimension
		# D_out is output dimension
		self.N = len(self.labels_o)#self.X_train.shape[0]
		try:
			self.D_in = len(self.labels_o[0])#self.X_train.shape[1]
		except IndexError:
			self.D_in = 1
		self.D_out = 1

		print('... Finished Init')
		sys.stdout.flush()

	def __call__(self,pixel_no):
		'''
		call instance so that train_pixel can be called with multiprocessing
		and still have all of the class instance variables

		:params pixel_no:
			Pixel number that is going to be trained

		'''
		print('3: {0}'.format(pixel_no))
		return self.train_pixel(pixel_no)


	def run(self,mp=False,ncpus=1):
		'''
		function to actually run the training on pixels

		:param mp (optional):
			boolean that turns on multiprocessing to run 
			training in parallel

		'''

		# initialize the output HDf5 file, return the datasets to populate
		outfile,wave_h5 = self.initout(restartfile=self.restartfile)

		# number of pixels to train
		numtrainedpixles = self.spectra.shape[0]
		print('... Number of Pixels in Spectrum: {0}'.format(numtrainedpixles))

		if self.restartfile == None:
			pixellist = range(numtrainedpixles)
		else:
			pixellist = list(np.argwhere(np.array(wave_h5) == 0.0).flatten())

		print('... Number of Pixels to Train: {0}'.format(len(pixellist)))
		sys.stdout.flush()

		# turn on multiprocessing if desired
		if mp:
			##### multiprocessing stuff #######
			try:
				# determine the number of cpu's and make sure we have access to them all
				numcpus = open('/proc/cpuinfo').read().count('processor\t:')
				os.system("taskset -p -c 0-{NCPUS} {PID}".format(NCPUS=numcpus-1,PID=os.getpid()))
			except IOError:
				pass

			pool = Pool(processes=ncpus)
			# init the map for the pixel training using the pool imap
			netout = pool.imap#(self,pixellist)

		else:
			# init the map for the pixel training using the standard serial imap
			netout = imap#(self,pixellist)

		# start total timer
		tottimestart = datetime.now()

		print('... Starting Training at {0}'.format(tottimestart))
		sys.stdout.flush()

		for pixellist_i in np.array_split(np.array(pixellist),int(numtrainedpixles/ncpus)):
			print('1: {0}'.format(pixellist_i))
			for ii,net in zip(pixellist_i,netout(self,pixellist_i)):
				print('2: {0}'.format(ii))
				wave_h5[ii]  = self.wavelength[ii]
				self.h5model_write(net[1],outfile,self.wavelength[ii])
				if self.saveopt:
					self.h5opt_write(net[2],outfile,self.wavelength[ii])
			# flush output file to save results
			sys.stdout.flush()
			outfile.flush()

		# print out total time
		print('Total time to train network: {0}'.format(datetime.now()-tottimestart))
		sys.stdout.flush()

		# formally close the output file
		outfile.close()

	def initout(self,restartfile=None):
		'''
		function to save all of the information into 
		a single HDF5 file
		'''

		if restartfile == None:
			# create output HDF5 file
			outfile = h5py.File(self.outfilename,'w')

			# add datesets for values that are already defined
			label_h5 = outfile.create_dataset('labels',    data=self.labels_o,  compression='gzip')
			resol_h5 = outfile.create_dataset('resolution',data=np.array([self.resolution]),compression='gzip')

			# define vectorized wavelength array, model array, and optimizer array
			wave_h5  = outfile.create_dataset('wavelength',data=np.zeros(len(self.wavelength)), compression='gzip')
			# model_h5 = outfile.create_dataset('model_arr', (len(self.wavelength),), compression='gzip')
			# opt_h5   = outfile.create_dataset('opt_arr', (len(self.wavelength),), compression='gzip')

			outfile.flush()

		else:
			# read in training file from restarted run
			outfile  = h5py.File(restartfile,'r+')

			# add datesets for values that are already defined
			label_h5 = outfile['labels']
			resol_h5 = outfile['resolution']

			# define vectorized arrays
			wave_h5  = outfile['wavelength']
			# model_h5 = outfile['model_arr']
			# opt_h5   = outfile['opt_arr']

		return outfile,wave_h5

	def train_pixel(self,pixel_no):
		'''
		define training function for each wavelength pixel to run in parallel
		note we create individual neural network for each pixel
		'''

		# start a timer
		starttime = datetime.now()
	
		# change labels into old_labels
		old_labels_o = self.labels_o

		# create tensor for labels
		X_train_Tensor = Variable(torch.from_numpy(old_labels_o).type(dtype))

		# pull fluxes at wavelength pixel
		Y_train = np.array(self.spectra[pixel_no,:]).T
		Y_train_Tensor = Variable(torch.from_numpy(Y_train).type(dtype), requires_grad=False)

		# initialize the model
		model = Net(self.D_in,self.H,self.D_out)

		# initialize the loss function
		loss_fn = torch.nn.MSELoss(size_average=False)
		# loss_fn = torch.nn.KLDivLoss(size_average=False)

		# initialize the optimizer
		learning_rate = 0.05
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		# optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

		# initialize the scheduler to adjust the learning rate
		scheduler = StepLR(optimizer,3,gamma=0.1)

		for epoch_i in range(self.epochs):
			# adjust the optimizer lr
			scheduler.step()
			lr_i = optimizer.param_groups[0]['lr']

			epochtime = datetime.now()

			for t in range(self.niter):
				steptime = datetime.now()
				def closure():
					# Before the backward pass, use the optimizer object to zero all of the
					# gradients for the variables it will update (which are the learnable weights
					# of the model)
					optimizer.zero_grad()

					# Forward pass: compute predicted y by passing x to the model.
					y_pred_train_Tensor = model(X_train_Tensor)

					# Compute and print loss.
					loss = loss_fn(y_pred_train_Tensor, Y_train_Tensor)

					# Backward pass: compute gradient of the loss with respect to model parameters
					loss.backward()
					
					if (t+1) % 100 == 0:
						print (
							'Pixel: {0} -- Step [{1:d}/{2:d}], Step Time: {3}, Loss: {4:.4f}'.format(
							pixel_no+1,t+1, self.niter, datetime.now()-steptime, loss.data[0])
						)
						sys.stdout.flush()

					return loss

				# Calling the step function on an Optimizer makes an update to its parameters
				optimizer.step(closure)

			# re-draw spectra for next epoch
			spectra_o,labels_o,wavelength = self.pullspectra(
				self.numtrain,resolution=self.resolution, waverange=self.waverange,
				MISTweighting=True,excludelabels=old_labels_o)
			spectra = spectra_o.T

			# create X tensor
			X_valid = labels_o
			X_valid_Tensor = Variable(torch.from_numpy(labels_o).type(dtype))

			# pull fluxes at wavelength pixel and create tensor
			Y_valid = np.array(spectra[pixel_no,:]).T
			Y_valid_Tensor = Variable(torch.from_numpy(Y_valid).type(dtype), requires_grad=False)
			
			# Validation Forward pass: compute predicted y by passing x to the model.
			Y_pred_valid_Tensor = model(X_valid_Tensor)
			Y_pred_valid = Y_pred_valid_Tensor.data.numpy()

			# calculate the residual at each validation label
			valid_residual = np.squeeze(Y_valid.T-Y_pred_valid.T)

			# create log of the validation step if user wants
			if self.logepoch:
				with open('ValidLog_pixel{0}_wave{1}_epoch{2}.log'.format(pixel_no+1,self.wavelength[pixel_no],epoch_i+1),'w') as logfile:
					logfile.write('valnum Teff log(g) [Fe/H] [a/Fe] resid\n')
					for ii,res in enumerate(valid_residual):
						logfile.write('{0} '.format(ii+1))
						logfile.write(np.array2string(X_valid[ii],separator=' ').replace('[','').replace(']',''))
						logfile.write(' {0}'.format(res))
						logfile.write('\n')

				fig = plt.figure()
				ax = fig.add_subplot(111)
				# residsize = ((10 * 
				# 	(max(np.abs(valid_residual))-np.abs(valid_residual))/
				# 	(max(np.abs(valid_residual))-min(np.abs(valid_residual)))
				# 	)**2.0) + 2.0

				residsize = ((150 * np.abs(valid_residual))**2.0) + 2.0

				scsym = ax.scatter(10.0**X_valid.T[0],X_valid.T[1],s=residsize,alpha=0.5)
				lgnd = ax.legend([scsym,scsym,scsym],
					# ['{0:5.3f}'.format(min(np.abs(valid_residual))),
					#  '{0:5.3f}'.format(np.median(np.abs(valid_residual))),
					#  '{0:5.3f}'.format(max(np.abs(valid_residual)))],
					['0.0','0.5','1.0'],
					 loc='upper left',
					)
				lgnd.legendHandles[0]._sizes = [2]
				lgnd.legendHandles[1]._sizes = [202]
				lgnd.legendHandles[2]._sizes = [402]
				# ax.invert_yaxis()
				# ax.invert_xaxis()
				ax.set_xlim(16000,3000)
				ax.set_ylim(6,-1.5)
				ax.set_xlabel('Teff')
				ax.set_ylabel('log(g)')
				fig.savefig('ValidLog_pixel{0}_wave{1}_epoch{2}.pdf'.format(pixel_no+1,self.wavelength[pixel_no],epoch_i+1))
				plt.close(fig)

			# check if user wants to do adaptive training
			if self.adaptivetrain:
				# sort validation labels on abs(resid)
				ind = np.argsort(np.abs(valid_residual))

				# determine worst 1% of validation set
				numbadmod = int(0.01*self.numtrain)
				# if number of bad models < 5, then by default set it to 5
				if numbadmod < 5:
					numbadmod = 5
				ind_s = ind[-numbadmod:]
				labels_a = labels_o[ind_s]

				# determine the number of models to add per new point
				numselmod = int(0.1*self.numtrain)
				# if number of new models < 5, then by default set it to 5
				if numselmod < 5:
					numselmod = 5

				numaddmod = numselmod/numbadmod
				if numaddmod <= 1:
					numaddmod = 2

				# make floor of numaddmod == 1
				# if numaddmod == 0:
				# 	numaddmod = 1

				# cycle through worst samples, adding 10% new models to training set
				for label_i in labels_a:
					while True:
						newlabels = np.array([x+(0.1*x)*np.random.randn(numaddmod) for x in label_i]).T
						spectra_ai,labels_ai,wavelength = self.pullspectra.selspectra(
							newlabels,
							resolution=self.resolution, 
							waverange=self.waverange,
							)
						# check to make sure labels_ai are unique
						if all([x_ai not in labels_o.tolist() for x_ai in labels_ai.tolist()]):
							break

					Y_valid_a = np.array(spectra_ai.T[pixel_no,:]).T
					Y_valid = np.hstack([Y_valid,Y_valid_a])
					labels_o = np.append(labels_o,labels_ai,axis=0)

				X_valid_Tensor = Variable(torch.from_numpy(labels_o).type(dtype))
				Y_valid_Tensor = Variable(torch.from_numpy(Y_valid).type(dtype), requires_grad=False)

			# re-use validation set as new training set for the next epoch
			old_labels_o = labels_o
			X_train_Tensor = X_valid_Tensor
			Y_train_Tensor = Y_valid_Tensor

			print (
				'Pixel: {0} -- EPOCH [{1:d}/{2:d}], Step Time: {3}, LR: {4}, med(|Res|): {5}'.format(
					pixel_no+1, epoch_i+1, self.epochs, datetime.now()-epochtime,
					lr_i,np.median(np.abs(valid_residual)))
				)
			sys.stdout.flush()			

		print('Trained pixel:{0}/{1} (wavelength: {2}), took: {3}'.format(
			pixel_no+1,len(self.spectra[:,0]),self.wavelength[pixel_no],
			datetime.now()-starttime))
		sys.stdout.flush()

		return [pixel_no, model, optimizer, datetime.now()-starttime]

	def h5model_write(self,model,th5,wavelength):
		'''
		Write trained model to HDF5 file
		'''
		try:
			for kk in model.state_dict().keys():
				th5.create_dataset('model_{0}/model/{1}'.format(wavelength,kk),
					data=model.state_dict()[kk].numpy(),
					compression='gzip')
		except RuntimeError:
			print('!!! PROBLEM WITH WRITING TO HDF5 FOR WAVELENGTH = {0} !!!'.format(wavelength))
			raise
		# th5.flush()

	def h5opt_write(self,optimizer,th5,wavelength):
		'''
		Write current state of the optimizer to HDF5 file
		'''
		for kk in optimizer.state_dict().keys():
			# cycle through the top keys
			if kk == 'state':
				# cycle through the different states
				for jj in optimizer.state_dict()['state'].keys():
					for ll in optimizer.state_dict()['state'][jj].keys():
	  					try:
							# check to see if it is a Tensor or an Int
							data = optimizer.state_dict()['state'][jj][ll].numpy()
						except AttributeError:
							# create an int array to save in the HDF5 file
							data = np.array([optimizer.state_dict()['state'][jj][ll]])
		          
						th5.create_dataset(
							'opt_{0}/optimizer/state/{1}/{2}'.format(wavelength,jj,ll),
							data=data,compression='gzip')
			elif kk == 'param_groups':
				pgdict = optimizer.state_dict()['param_groups'][0]
				for jj in pgdict.keys():
					try:
						th5.create_dataset(
							'opt_{0}/optimizer/param_groups/{1}'.format(wavelength,jj),
							data=np.array(pgdict[jj]),compression='gzip')
					except TypeError:
						th5.create_dataset(
							'opt_{0}/optimizer/param_groups/{1}'.format(wavelength,jj),
							data=np.array([pgdict[jj]]),compression='gzip')

		# th5.flush()
