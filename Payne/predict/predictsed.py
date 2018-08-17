#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .photANN import ANN, fastANN

class PayneSEDPredict(object):

    def __init__(self, usebands=None, nnpath=None):
        self.anns = self._initphotnn(usebands,nnpath=nnpath)

    def _initphotnn(self, filterarray, nnpath=None):
        ANNdict = {}
        for ff in filterarray:
            try:
                ANNdict[ff] = ANN(ff, nnpath=nnpath, verbose=False)
            except IOError:
                print('Cannot find NN HDF5 file for {0}'.format(ff))
        return ANNdict

    def sed(self, logt=None, logg=None, feh=None,
            logl=None, av=0.0, dist=None, logA=None, filters=None):
        """
        """

        if type(filters) == type(None):
        	filters = self.anns.keys()
        BC = np.array([self.anns[f].eval([10**logt, logg, feh, av])
                       for f in filters])
        if (type(logl) != type(None)) and (type(dist) != type(None)):
            mu = 5 * np.log10(dist) - 5
            m = -2.5 * logl + 4.74 - BC + mu
        elif (type(logA) != type(None)):
            m = 5.0*logA - 10.0*(logt - np.log10(5770.0)) - 0.26 - BC
        else:
            raise IOError('cannot understand input pars into sed function')
        return m

class FastPayneSEDPredict(object):
    
    def __init__(self, usebands=None, nnpath=None):
        self.filternames = usebands
        nnlist = [ANN(f, nnpath=nnpath, verbose=False) for f in usebands]
        self.anns = fastANN(nnlist, self.filternames)

    def sed(self, logt=None, logg=None, feh=None,
            logl=None, av=0.0, dist=None, logA=None, band_indices=slice(None)):
        """
        """
        BC = self.anns.eval([10**logt, logg, feh, av])
        
        if (type(logl) != type(None)) and (type(dist) != type(None)):
            mu = 5.0 * np.log10(dist) - 5.0
            m = -2.5 * logl + 4.74 - BC + mu
        elif (type(logA) != type(None)):
            m = 5.0*logA - 10.0*(logt - np.log10(5770.0)) - 0.26 - BC
        else:
            raise IOError('cannot understand input pars into sed function')


        try:
            return m[band_indices]
        except IndexError:
            return [m]