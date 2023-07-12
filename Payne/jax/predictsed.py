#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as np
from jax import lax
from .photANN import ANN, fastANN
from .highred import highAv
import glob

class PayneSEDPredict(object):

    def __init__(self, usebands=None, nnpath=None):
        self.anns = self._initphotnn(usebands,nnpath=nnpath)

    def _initphotnn(self, usebands=None, nnpath=None):
        if usebands == None:
            # user doesn't know which filters, so read in all that
            # are contained in photNN path
            flist = glob.glob(nnpath+'/nn*h5')
            allfilters = [x.split('/')[-1].replace('nnMIST_','').replace('.h5','') for x in flist]
            usebands = allfilters
        self.filternames = usebands        

        ANNdict = {}
        for ff in self.filternames:
            try:
                ANNdict[ff] = ANN(ff, nnpath=nnpath, verbose=False)
            except IOError:
                print('Cannot find NN HDF5 file for {0}'.format(ff))
        return ANNdict

    def sed(self, logt=None, logg=None, feh=None, afe=None,
            logl=None, av=0.0, rv=None,
            dist=None, logA=None, filters=None):
        """
        """

        if type(filters) == type(None):
            filters = self.anns.keys()
            
        if type(rv) == type(None):
            inpars = [10.0**logt,logg,feh,afe,av]
        else:
            inpars = [10.0**logt,logg,feh,afe,av,rv]

        BC = np.array([self.anns[f].eval(inpars)
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
        if usebands == None:
            # user doesn't know which filters, so read in all that
            # are contained in photNN path
            flist = glob.glob(nnpath+'/nnMIST_*.h5')
            allfilters = [x.split('/')[-1].replace('nnMIST_','').replace('.h5','') for x in flist]
            usebands = allfilters
        self.filternames = usebands    

        nnlist = []
        for f in usebands:
            try:
                nnlist.append(ANN(f, nnpath=nnpath, verbose=False))
            except:
                pass
        self.anns = fastANN(nnlist, self.filternames)

        self.HiAv = highAv(self.filternames)

    def sed(self, logt=None, logg=None, feh=None, afe=None,
            logl=None, av=0.0, rv=3.1, 
            dist=None, logA=None, band_indices=slice(None)):
        """
        """

        # if type(rv) == type(None):
        #     inpars = [10.0**logt,logg,feh,afe,av]
        # else:
        inpars = [10.0**logt,logg,feh,afe,av,rv]

        def bcdefault(x):
            return self.anns.eval(inpars)

        def bchiav(x):
            BC0 = self.anns.eval([10.0**logt,logg,feh,afe,0.0,3.1])
            return self.HiAv.calc(BC0,av,rv)
        
        BC = lax.cond(av < 5.0,bcdefault,bchiav,None)

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