# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import h5py
from scipy import constants
speedoflight = constants.c / 1000.0
from scipy import stats

from ..predict import predictspec
from ..predict import ystpred
from ..utils.quantiles import quantile
from ..utils.readc3k import readc3k

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
	def __init__(self, NNfilename, NNtype='LinNet',c3kpath=None,ystnn=None):
		# user inputed neural-net output file
		self.NNfilename = NNfilename

		# type of NN
		self.NNtype = NNtype

		# initialize the Payne Predictor
		self.NN = predictspec.ANN(
			nnpath=self.NNfilename,
			NNtype=self.NNtype,
			testing=True,
			verbose=True)

		# define wavelengths
		self.wave = self.NN.wavelength

		# default resolution is the native resolution of NN
		self.resolution = self.NN.resolution

		if c3kpath == None:
			self.c3kpath = '/Users/pcargile/Astro/ThePayne/train_grid/rv31/grid/'
		else:
			self.c3kpath = c3kpath

		if ystnn == None:
			self.ystnn = None#'/Users/pcargile/Astro/ThePayne/YSdata/YSTANN_wvt.h5'
		else:
			self.ystnn = ystnn

	def runtest(self,**kwargs):
		'''
		Function to run the testing on an already trained network
		'''
		# output plot file
		output = kwargs.get('output','./test.pdf')

		if 'testnum' in kwargs:
			testnum = kwargs['testnum']
			ind = rng.integers(low=0,high=len(self.NN.testlabels),size=testnum)
			testpred = self.NN.testpred[ind]
			testlabels = self.NN.testlabels[ind]
		else:
			# test with equal number of spectra as training
			testnum = len(self.NN.testlabels)
			testpred = self.NN.testpred
			testlabels = self.NN.testlabels

		# make NN predictions for all test labels
		nnpred = np.array([self.NN.eval(pars) for pars in testlabels])

		# make residual array
		modres = np.array([np.abs(x-y) for x,y in zip(testpred,nnpred)])

		# initialize PDF file
		with PdfPages(output) as pdf:
			# histxrange = np.log10(quantile(np.median(modres,axis=0),[0.0001,0.9999]))
			histxrange = [-4.5,-1]
			#axis=0 -> median at each pixel
			#axis=1 -> median of each spectrum

			# build MAD versus lambda plots
			fig,ax = plt.subplots(nrows=2,ncols=1,constrained_layout=True,figsize=(8,8))

			ax[0].scatter(self.wave,np.log10(np.median(modres,axis=0))
				,marker='.',s=5,ec='none')
			ax[1].hist(np.log10(np.median(modres,axis=0)),
				bins=50,cumulative=True,density=True,
				range=histxrange,histtype='step')

			ax[0].set_xlabel(r'$\lambda$')
			ax[0].set_ylabel('log(median MAD @ pixel)')

			ax[1].set_xlabel('log(median MAD @ pixel)')
			ax[1].set_ylabel('CDF (% of pixels)')

			pdf.savefig(fig)
			plt.close(fig)

			# MAD histograms binned by pars
			fig,ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True,figsize=(8,8))

			# teff binned
			ind = testlabels[...,0] > 6500.0
			ax[0,0].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C0',range=histxrange,
				histtype='step',label='Teff > 6500',
				density=False,lw=2.0,alpha=0.75)
			ind = (testlabels[...,0] <= 6500.0) & (testlabels[...,0] > 4500.0)
			ax[0,0].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C3',range=histxrange,
				histtype='step',label='4500 < Teff < 6500',
				density=False,lw=2.0,alpha=0.75)
			ind = (testlabels[...,0] <= 4500.0) 
			ax[0,0].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C4',range=histxrange,
				histtype='step',label='Teff < 4500',
				density=False,lw=2.0,alpha=0.75)
			# ind = (testlabels[...,0] <= 4500.0) 
			# ax[0,0].hist(np.log10(np.median(modres[ind],axis=1)),
			# 	bins=25,color='C5',range=histxrange,
			# 	histtype='step',label='Teff < 4500',
			# 	density=False)
			ax[0,0].legend(fontsize=7,frameon=False)

			# logg binned
			ind = testlabels[...,1] > 4.0
			ax[0,1].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C0',range=histxrange,
				histtype='step',label='log(g) > 4.0',
				density=False,lw=2.0,alpha=0.75)
			ind = (testlabels[...,1] <= 4.0) & (testlabels[...,1] > 3.0)
			ax[0,1].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C3',range=histxrange,
				histtype='step',label='3.0 < log(g) < 4.0',
				density=False,lw=2.0,alpha=0.75)
			ind = (testlabels[...,1] <= 3.0) 
			ax[0,1].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C4',range=histxrange,
				histtype='step',label='log(g) < 3.0',
				density=False,lw=2.0,alpha=0.75)
			# ind = (testlabels[...,1] <= 3.5) 
			# ax[0,1].hist(np.log10(np.median(modres[ind],axis=1)),
			# 	bins=25,color='C5',range=histxrange,
			# 	histtype='step',label='log(g) < 3.5',
			# 	density=True)
			ax[0,1].legend(fontsize=7,frameon=False)

			# feh binned
			ind = testlabels[...,2] > 0.0
			ax[1,0].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C0',range=histxrange,
				histtype='step',label=' [Fe/H] > 0.0',
				density=False,lw=2.0,alpha=0.75)
			ind = (testlabels[...,2] <= 0.0) & (testlabels[...,2] > -1.0)
			ax[1,0].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C3',range=histxrange,
				histtype='step',label='-1.0 < [Fe/H] < 0.0',
				density=False,lw=2.0,alpha=0.75)
			ind = (testlabels[...,2] <= -1.0)
			ax[1,0].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C4',range=histxrange,
				histtype='step',label='[Fe/H] < -1.0',
				density=False,lw=2.0,alpha=0.75)
			# ind = (testlabels[...,2] <= -1.5) 
			# ax[1,0].hist(np.log10(np.median(modres[ind],axis=1)),
			# 	bins=25,color='C5',range=histxrange,
			# 	histtype='step',label=' [Fe/H] < -1.5',
			# 	density=False,lw=2.0,alpha=0.75)
			ax[1,0].legend(fontsize=7,frameon=False)

			# afe binned
			ind = testlabels[...,3] > 0.3
			ax[1,1].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C0',range=histxrange,
				histtype='step',label=' [a/Fe] > 0.3',
				density=False,lw=2.0,alpha=0.75)
			ind = (testlabels[...,3] <= 0.3) & (testlabels[...,3] > 0.0)
			ax[1,1].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C3',range=histxrange,
				histtype='step',label=' 0.0 < [a/Fe] < 0.3',
				density=False,lw=2.0,alpha=0.75)
			ind = (testlabels[...,3] <= 0.0)
			ax[1,1].hist(np.log10(np.median(modres[ind],axis=1)),
				bins=25,color='C4',range=histxrange,
				histtype='step',label=' [a/Fe] < 0.0',
				density=False,lw=2.0,alpha=0.75)
			# ind = (testlabels[...,3] <= 0.0) 
			# ax[1,1].hist(np.log10(np.median(modres[ind],axis=1)),
			# 	bins=25,color='C5',range=histxrange,
			# 	histtype='step',label=' [a/Fe] < 0.0',
			# 	density=True)
			ax[1,1].legend(fontsize=7,frameon=False)

			ax[1,0].set_xlabel('log median MAD per spectrum')
			ax[1,1].set_xlabel('log median MAD per spectrum')

			pdf.savefig(fig)
			plt.close(fig)


			# build MAD versus lambda binned by pars 2x2
			fig,ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True,figsize=(8,8))

			# teff binned
			ind = testlabels[...,0] > 6500.0
			ax[0,0].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C0',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[0,0].plot(bin_centers,bin_med,lw=1.0,c='C0',label='Teff > 6500')

			ind = (testlabels[...,0] <= 6500.0) & (testlabels[...,0] > 4500.0)
			ax[0,0].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C3',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[0,0].plot(bin_centers,bin_med,lw=1.0,c='C3',label='4500 < Teff < 6500')

			ind = (testlabels[...,0] <= 4500.0)
			ax[0,0].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C4',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[0,0].plot(bin_centers,bin_med,lw=1.0,c='C4',label='Teff < 4500')
			ax[0,0].legend(fontsize=7,frameon=False)

			# ind = (testlabels[...,0] <= 4500.0) 
			# ax[0,0].scatter(self.wave,np.median(modres[ind],axis=0),marker='.',c='C5')

			# logg binned
			ind = testlabels[...,1] > 4.0
			ax[0,1].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C0',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[0,1].plot(bin_centers,bin_med,lw=1.0,c='C0',label='log(g) > 4.0')

			ind = (testlabels[...,1] <= 4.0) & (testlabels[...,1] > 3.0)
			ax[0,1].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C3',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[0,1].plot(bin_centers,bin_med,lw=1.0,c='C3',label='3.0 < log(g) < 4.0')

			ind = (testlabels[...,1] <= 3.0) 
			ax[0,1].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C4',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[0,1].plot(bin_centers,bin_med,lw=1.0,c='C4',label='log(g) < 3.0')
			ax[0,1].legend(fontsize=7,frameon=False)
			# ind = (testlabels[...,1] <= 3.5) 
			# ax[0,1].scatter(self.wave,np.median(modres[ind],axis=0),marker='.',c='C5')

			# feh binned
			ind = testlabels[...,2] > 0.0
			ax[1,0].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C0',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[1,0].plot(bin_centers,bin_med,lw=1.0,c='C0',label='[Fe/H] > 0.0')

			ind = (testlabels[...,2] <= 0.0) & (testlabels[...,2] > -1.0)
			ax[1,0].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C3',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[1,0].plot(bin_centers,bin_med,lw=1.0,c='C3',label='-1.0 < [Fe/H] < 0.0')

			ind = (testlabels[...,2] <= -1.0) 
			ax[1,0].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C4',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[1,0].plot(bin_centers,bin_med,lw=1.0,c='C4',label='[Fe/H] < -1.0')
			ax[1,0].legend(fontsize=7,frameon=False)
			# ind = (testlabels[...,2] <= -1.5) 
			# ax[1,0].scatter(self.wave,np.median(modres[ind],axis=0),marker='.',c='C5')

			# afe binned
			ind = testlabels[...,3] > 0.3
			ax[1,1].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C0',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[1,1].plot(bin_centers,bin_med,lw=1.0,c='C0',label='[a/Fe] > 0.3')

			ind = (testlabels[...,3] <= 0.3) & (testlabels[...,3] > 0.0)
			ax[1,1].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C3',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[1,1].plot(bin_centers,bin_med,lw=1.0,c='C3',label='0.0 < [a/Fe] < 0.3')

			ind = (testlabels[...,3] <= 0.0) 
			ax[1,1].scatter(self.wave,np.log10(np.median(modres[ind],axis=0)),
				marker='.',c='C4',s=1,alpha=0.75,ec='none')
			bin_med, bin_edges, binnumber = stats.binned_statistic(
				self.wave,np.log10(np.median(modres[ind],axis=0)), 
				statistic='median', bins=25)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax[1,1].plot(bin_centers,bin_med,lw=1.0,c='C4',label='[a/Fe] < 0.0')
			ax[1,1].legend(fontsize=7,frameon=False)
			# ind = (testlabels[...,3] <= 0.0) 
			# ax[1,1].scatter(self.wave,np.median(modres[ind],axis=0),marker='.',c='C5')

			ax[0,0].set_ylim(-4.5,-1.0)
			ax[1,0].set_ylim(-4.5,-1.0)
			ax[0,1].set_ylim(-4.5,-1.0)
			ax[1,1].set_ylim(-4.5,-1.0)

			ax[1,0].set_xlabel(r'$\lambda$')
			ax[1,1].set_xlabel(r'$\lambda$')
			ax[0,0].set_ylabel('log(median MAD @ pixel)')
			ax[1,0].set_ylabel('log(median MAD @ pixel)')

			pdf.savefig(fig)
			plt.close(fig)

			# build MAD map for 4 different pars

			# build MAD CDF comparison to YST model
			inpars = [5770.0,4.4,0.0,0.0,1.0]

			NN = predictspec.ANN(nnpath=self.NNfilename,NNtype=self.NNtype,verbose=False)
			if self.ystnn == None:
				NN_yst = np.nan
			else:
				NN_yst = ystpred.Net(self.ystnn)

			c3kmods = readc3k(MISTpath=None,C3Kpath=self.c3kpath,vtfixed=True)
			out = c3kmods.selspectra(inpars,resolution=self.resolution,waverange=[self.wave.min(),self.wave.max()])

			pars = list(out[1][0])[:len(testlabels[0])]
			flux = out[0][0]
			wave = out[2]

			modflux = np.interp(wave,NN.wavelength,NN.eval(pars))
			if self.ystnn == None:
				modflux_yst = [np.nan for _ in range(len(wave))]
			else:
				modflux_yst = np.interp(wave,NN_yst.wavelength,NN_yst.eval(pars))

			fig,ax = plt.subplots(nrows=5,ncols=1,constrained_layout=True,figsize=(8,8))

			ax[0].plot(wave,modflux,lw=1.0,c='C0',label='New NN')
			ax[0].plot(wave,modflux_yst,lw=1.0,c='C1',label='YST NN')
			ax[0].plot(wave,flux,lw=0.25,c='k',label='C3K')
			ax[0].set_xlim(wave.min(),wave.max())

			ax[1].plot(wave,modflux-flux,    lw=1.0,c='C0',alpha=0.5)
			ax[1].plot(wave,modflux_yst-flux,lw=1.0,c='C1',alpha=0.5)
			ax[1].set_xlim(wave.min(),wave.max())

			ax[2].plot(wave,modflux,lw=1.0,c='C0',label='New NN')
			ax[2].plot(wave,modflux_yst,lw=1.0,c='C1',label='YST NN')
			ax[2].plot(wave,flux,lw=0.25,c='k',label='C3K')
			ax[2].set_xlim(8425,8475)

			ax[3].plot(wave,modflux,lw=1.0,c='C0',label='New NN')
			ax[3].plot(wave,modflux_yst,lw=1.0,c='C1',label='YST NN')
			ax[3].plot(wave,flux,lw=0.25,c='k',label='C3K')
			ax[3].set_xlim(8650,8700)

			parstr = (
					'Teff = {TEFF:.0f}\n'+
					'log(g) = {LOGG:.2f}\n'+
					'[Fe/H] = {FEH:.2f}\n'+
					'[a/Fe] = {AFE:.2f}'
				).format(
				TEFF=pars[0],
				LOGG=pars[1],
				FEH=pars[2],
				AFE=pars[3]
				)

			ax[4].text(
				0.85,0.05,
				parstr,
				horizontalalignment='left',
				verticalalignment='bottom', 
				transform=ax[4].transAxes)

			ax[4].hist(np.log10(np.abs(modflux-flux)),lw=1.0,color='C0',
				bins=50,cumulative=True,density=True,
				histtype='step',range=[-4.5,-1])

			ax[4].hist(np.log10(np.abs(modflux_yst-flux)),lw=1.0,color='C1',
				bins=50,cumulative=True,density=True,
				histtype='step',range=[-4.5,-1])
			ax[4].hlines(
				y=stats.percentileofscore(np.log10(np.abs(modflux-flux)),-2.0)/100.0,
				xmin=-4.5,xmax=-2.0,
				colors='C0',alpha=0.8)
			ax[4].hlines(
				y=stats.percentileofscore(np.log10(np.abs(modflux_yst-flux)),-2.0)/100.0,
				xmin=-4.5,xmax=-2.0,
				colors='C1',alpha=0.8)

			ax[4].set_xlim(-4.5,-1.0)
			ax[4].set_xlabel('log Abs Deviation')
			ax[4].set_ylabel('CDF [%]')

			pdf.savefig(fig)
			plt.close(fig)

			inpars = [4000.0,2.5,0.0,0.0,1.0]
			out = c3kmods.selspectra(inpars,resolution=self.resolution,waverange=[self.wave.min(),self.wave.max()])

			pars = list(out[1][0])[:len(testlabels[0])]
			flux = out[0][0]
			wave = out[2]

			modflux = np.interp(wave,NN.wavelength,NN.eval(pars))
			if self.ystnn == None:
				modflux_yst = [np.nan for _ in range(len(wave))]
			else:
				modflux_yst = np.interp(wave,NN_yst.wavelength,NN_yst.eval(pars))

			fig,ax = plt.subplots(nrows=5,ncols=1,constrained_layout=True,figsize=(8,8))

			ax[0].plot(wave,modflux,lw=1.0,c='C0',label='New NN')
			ax[0].plot(wave,modflux_yst,lw=1.0,c='C1',label='YST NN')
			ax[0].plot(wave,flux,lw=0.25,c='k',label='C3K')
			ax[0].set_xlim(wave.min(),wave.max())

			ax[1].plot(wave,modflux-flux,    lw=1.0,c='C0',alpha=0.5)
			ax[1].plot(wave,modflux_yst-flux,lw=1.0,c='C1',alpha=0.5)
			ax[1].set_xlim(wave.min(),wave.max())

			ax[2].plot(wave,modflux,lw=1.0,c='C0',label='New NN')
			ax[2].plot(wave,modflux_yst,lw=1.0,c='C1',label='YST NN')
			ax[2].plot(wave,flux,lw=0.25,c='k',label='C3K')
			ax[2].set_xlim(8425,8475)

			ax[3].plot(wave,modflux,lw=1.0,c='C0',label='New NN')
			ax[3].plot(wave,modflux_yst,lw=1.0,c='C1',label='YST NN')
			ax[3].plot(wave,flux,lw=0.25,c='k',label='C3K')
			ax[3].set_xlim(8650,8700)

			parstr = (
					'Teff = {TEFF:.0f}\n'+
					'log(g) = {LOGG:.2f}\n'+
					'[Fe/H] = {FEH:.2f}\n'+
					'[a/Fe] = {AFE:.2f}'
				).format(
				TEFF=pars[0],
				LOGG=pars[1],
				FEH=pars[2],
				AFE=pars[3]
				)

			ax[4].text(
				0.85,0.05,
				parstr,
				horizontalalignment='left',
				verticalalignment='bottom', 
				transform=ax[4].transAxes)

			ax[4].hist(np.log10(np.abs(modflux-flux)),lw=1.0,color='C0',
				bins=50,cumulative=True,density=True,
				histtype='step',range=[-4.5,-1])

			ax[4].hist(np.log10(np.abs(modflux_yst-flux)),lw=1.0,color='C1',
				bins=50,cumulative=True,density=True,
				histtype='step',range=[-4.5,-1])

			ax[4].hlines(
				y=stats.percentileofscore(np.log10(np.abs(modflux-flux)),-2.0)/100.0,
				xmin=-4.5,xmax=-2.0,
				colors='C0',alpha=0.8)
			ax[4].hlines(
				y=stats.percentileofscore(np.log10(np.abs(modflux_yst-flux)),-2.0)/100.0,
				xmin=-4.5,xmax=-2.0,
				colors='C1',alpha=0.8)

			ax[4].set_xlim(-4.5,-1.0)
			ax[4].set_xlabel('log Abs Deviation')
			ax[4].set_ylabel('CDF [%]')

			pdf.savefig(fig)
			plt.close(fig)

			inpars = [4500.0,5.0,0.0,0.0,1.0]
			out = c3kmods.selspectra(inpars,resolution=self.resolution,waverange=[self.wave.min(),self.wave.max()])

			pars = list(out[1][0])[:len(testlabels[0])]
			flux = out[0][0]
			wave = out[2]

			modflux = np.interp(wave,NN.wavelength,NN.eval(pars))
			if self.ystnn == None:
				modflux_yst = [np.nan for _ in range(len(wave))]
			else:
				modflux_yst = np.interp(wave,NN_yst.wavelength,NN_yst.eval(pars))

			fig,ax = plt.subplots(nrows=5,ncols=1,constrained_layout=True,figsize=(8,8))

			ax[0].plot(wave,modflux,lw=1.0,c='C0',label='New NN')
			ax[0].plot(wave,modflux_yst,lw=1.0,c='C1',label='YST NN')
			ax[0].plot(wave,flux,lw=0.25,c='k',label='C3K')
			ax[0].set_xlim(wave.min(),wave.max())

			ax[1].plot(wave,modflux-flux,    lw=1.0,c='C0',alpha=0.5)
			ax[1].plot(wave,modflux_yst-flux,lw=1.0,c='C1',alpha=0.5)
			ax[1].set_xlim(wave.min(),wave.max())

			ax[2].plot(wave,modflux,lw=1.0,c='C0',label='New NN')
			ax[2].plot(wave,modflux_yst,lw=1.0,c='C1',label='YST NN')
			ax[2].plot(wave,flux,lw=0.25,c='k',label='C3K')
			ax[2].set_xlim(8425,8475)

			ax[3].plot(wave,modflux,lw=1.0,c='C0',label='New NN')
			ax[3].plot(wave,modflux_yst,lw=1.0,c='C1',label='YST NN')
			ax[3].plot(wave,flux,lw=0.25,c='k',label='C3K')
			ax[3].set_xlim(8650,8700)

			parstr = (
					'Teff = {TEFF:.0f}\n'+
					'log(g) = {LOGG:.2f}\n'+
					'[Fe/H] = {FEH:.2f}\n'+
					'[a/Fe] = {AFE:.2f}'
				).format(
				TEFF=pars[0],
				LOGG=pars[1],
				FEH=pars[2],
				AFE=pars[3]
				)

			ax[4].text(
				0.85,0.05,
				parstr,
				horizontalalignment='left',
				verticalalignment='bottom', 
				transform=ax[4].transAxes)

			ax[4].hist(np.log10(np.abs(modflux-flux)),lw=1.0,color='C0',
				bins=50,cumulative=True,density=True,
				histtype='step',range=[-4.5,-1])

			ax[4].hist(np.log10(np.abs(modflux_yst-flux)),lw=1.0,color='C1',
				bins=50,cumulative=True,density=True,
				histtype='step',range=[-4.5,-1])

			ax[4].hlines(
				y=stats.percentileofscore(np.log10(np.abs(modflux-flux)),-2.0)/100.0,
				xmin=-4.5,xmax=-2.0,
				colors='C0',alpha=0.8)
			ax[4].hlines(
				y=stats.percentileofscore(np.log10(np.abs(modflux_yst-flux)),-2.0)/100.0,
				xmin=-4.5,xmax=-2.0,
				colors='C1',alpha=0.8)

			ax[4].set_xlim(-4.5,-1.0)
			ax[4].set_xlabel('log Abs Deviation')
			ax[4].set_ylabel('CDF [%]')

			pdf.savefig(fig)
			plt.close(fig)

			# build MAD for standard stars (Sun, Arcturus, Procyon, 61 CygA)

			# fig,ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True,figsize=(8,8))

			# pdf.savefig(fig)
			# plt.close(fig)
