#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as np
from jax import lax
from .photANN import ANN, fastANN
from .highred import highAv

_ALLFILTERS = (
    ['2MASS_H', '2MASS_J', '2MASS_Ks', 
    'Bessell_B', 'Bessell_I', 'Bessell_R', 'Bessell_U', 'Bessell_V', 
    'DECam_g', 'DECam_i', 'DECam_r', 'DECam_u', 'DECam_Y', 'DECam_z', 
    'GaiaMAW_BPb', 'GaiaMAW_BPf','GaiaMAW_G', 'GaiaMAW_RP', 
    # 'GALEX_FUV', 'GALEX_NUV', 
    'Hipparcos_Hp', 
    'Kepler_D51', 'Kepler_Kp', 
    'PS_g', 'PS_i', 'PS_open', 'PS_r', 'PS_w', 'PS_y', 'PS_z', 
    'SDSS_g', 'SDSS_i', 'SDSS_r', 'SDSS_u', 'SDSS_z', 
    'TESS_T', 
    'Tycho_B', 'Tycho_V', 
    'UKIDSS_H', 'UKIDSS_J', 'UKIDSS_K', 'UKIDSS_Y', 'UKIDSS_Z', 
    'WISE_W1', 'WISE_W2', 'WISE_W3', 'WISE_W4']
    )

class PayneSEDPredict(object):

    def __init__(self, usebands=None, nnpath=None):
        self.anns = self._initphotnn(usebands,nnpath=nnpath)

    def _initphotnn(self, usebands=None, nnpath=None):
        if usebands == None:
            usebands = _ALLFILTERS
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
            usebands = _ALLFILTERS
        self.filternames = usebands        
        nnlist = [ANN(f, nnpath=nnpath, verbose=False) for f in usebands]
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