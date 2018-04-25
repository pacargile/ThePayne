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
            logl=0.0, av=0.0, dist=10.0, filters=None):
        """
        """

        if type(filters) == type(None):
        	filters = self.anns.keys()
        mu = 5 * np.log10(dist) - 5
        BC = np.array([self.anns[f].eval([10**logt, logg, feh, av])
                       for f in filters])

        m = -2.5 * logl + 4.74 - BC + mu

        return m

class FastPayneSEDPredict(object):
    
    def __init__(self, usebands=None, nnpath=None):
        self.filternames = usebands
        nnlist = [ANN(f, nnpath=nnpath, verbose=False) for f in usebands]
        self.anns = fastANN(nnlist, self.filternames)

    def sed(self, logt=None, logg=None, feh=None,
            logl=0.0, av=0.0, dist=10.0, band_indices=slice(None)):
        """
        """
        mu = 5.0 * np.log10(dist) - 5.0
        BC = self.anns.eval([10**logt, logg, feh, av])

        m = -2.5 * logl + 4.74 - BC + mu

        try:
            return m[band_indices]
        except IndexError:
            return [m]