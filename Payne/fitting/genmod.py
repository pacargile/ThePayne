# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime

from .fitutils import polycalc

class GenMod(object):
    """docstring for GenMod"""
    def __init__(self, *arg, **kwargs):
        super(GenMod, self).__init__()
        self.verbose = kwargs.get('verbose',False)
        
    def _initspecnn(self,nnpath=None,**kwargs):
        from .fitutils import polycalc
        
        NNtype = kwargs.get('NNtype','YST1')

        if NNtype == 'YST1':
            from Payne.predict.ystpred import PayneSpecPredict
        else:
            from Payne.predict.predictspec import PayneSpecPredict

        carbon_bool = kwargs.get('carbon_bool',False)

        # initialize the Payne Spectrum Predictor
        Cnnpath = kwargs.get('Cnnpath',None)
        if Cnnpath is None:
            self.PP = PayneSpecPredict(nnpath=nnpath,NNtype=NNtype)
        else:
            self.PP = PayneSpecPredict(nnpath=nnpath,Cnnpath=Cnnpath,NNtype=NNtype)


    def _initphotnn(self,filterarray,nnpath=None):
        self.filterarray = filterarray

        from ..predict.predictsed import FastPayneSEDPredict
        self.fppsed = FastPayneSEDPredict(
            usebands=self.filterarray,nnpath=nnpath,
            )
        if self.filterarray is None:
            self.filterarray = self.fppsed.filternames


        # from ..predict.photANN import ANN


        # # for each input filter, read in the ANN file
        # ANNdict = {}
        # for ff in self.filterarray:
        #   try:
        #       ANNdict[ff] = ANN(ff,nnpath=nnpath,verbose=self.verbose)
        #   except IOError:
        #       print('Cannot find NN HDF5 file for {0}'.format(ff))
        # self.ANNdict = ANNdict

    def genspec(self,pars,outwave=None,verbose=False,
        modpoly=False,carbon_bool=False):
        # define parameters from pars array
        Teff = pars[0]
        logg = pars[1]
        FeH  = pars[2]
        aFe = pars[3]
        radvel = pars[4]
        rotvel = pars[5]
        vmic = pars[6]
        inst_R = pars[7]

        # check to see if a polynomial is used for spectrum normalization
        if modpoly and not carbon_bool:
            polycoef = pars[8:]
        else:
            polycoef = pars[8:-1]

        if carbon_bool:
            CarbonScale = pars[-1]
        else:
            CarbonScale = 0.0

        # check to see if inst_R is a float or an array of dispersions
        if isinstance(inst_R,float):
            input_inst_R = 2.355 * inst_R
        else:
            input_inst_R = inst_R

        if verbose:
            print('Teff   = {}'.format(Teff))
            print('log(g) = {}'.format(logg))
            print('[Fe/H] = {}'.format(FeH))
            print('[a/Fe] = {}'.format(aFe))
            print('RV     = {}'.format(radvel))
            print('Vstar  = {}'.format(rotvel))
            print('Vmic   = {}'.format(vmic))
            print('InstR  = {}'.format(input_inst_R))

        # predict model flux at model wavelengths
        modwave_i,modflux_i = self.PP.getspec(
            Teff=Teff,logg=logg,feh=FeH,afe=aFe,rad_vel=radvel,rot_vel=rotvel,
            vmic=vmic,inst_R=input_inst_R,
            outwave=outwave)       
        # if polynomial normalization is turned on then multiply model by it
        if modpoly:
            epoly = polycalc(polycoef,modwave_i)
            # now multiply the model by the polynomial normalization poly
            modflux_i = modflux_i*epoly

        return modwave_i,modflux_i

    def genphot(self,pars,rvfree=False,verbose=False):
        # define parameters from pars array
        Teff = pars[0]
        logg = pars[1]
        FeH  = pars[2]
        aFe  = pars[3]
        logR = pars[4]
        Dist = pars[5]
        Av   = pars[6]
        if rvfree:
            Rv = pars[7]
        else:
            Rv = 3.1

        logTeff = np.log10(Teff)

        logL = 2.0*logR + 4.0*(logTeff - np.log10(5770.0))

        # create parameter dictionary
        photpars = {}
        photpars['logt'] = logTeff
        photpars['logg'] = logg
        photpars['feh']  = FeH
        photpars['afe']  = aFe
        photpars['logl'] = logL
        photpars['dist'] = Dist
        photpars['av']   = Av
        photpars['rv']   = Rv

        # create filter list and arrange photometry to this list

        # sed = self.ppsed.sed(filters=filterlist,**photpars)
        sed = self.fppsed.sed(**photpars)

        outdict = {ff_i:sed_i for sed_i,ff_i in zip(sed,self.filterarray)}

        # # calculate absolute bolometric magnitude
        # Mbol = -2.5*(2.0*logR + 4.0*np.log10(Teff/5770.0)) + 4.74

        # # calculate BC for all photometry in obs_phot dict
        # outdict = {}
        # for ii,kk in enumerate(self.filterarray):
        #   BCdict_i = float(self.ANNdict[kk].eval([Teff,logg,FeH,Av]))
        #   outdict[kk] = (Mbol - BCdict_i) + 5.0*np.log10(Dist) - 5.0

        return outdict

    def genphot_scaled(self,pars,rvfree=False,verbose=False):
        # define parameters from pars array
        Teff = pars[0]
        logg = pars[1]
        FeH  = pars[2]
        aFe  = pars[3]
        logA = pars[4]
        Av   = pars[5]
        if rvfree:
            Rv = pars[6]
        else:
            Rv = 3.1

        logTeff = np.log10(Teff)

        # create parameter dictionary
        photpars = {}
        photpars['logt'] = logTeff
        photpars['logg'] = logg
        photpars['feh']  = FeH
        photpars['afe']  = aFe
        photpars['logA'] = logA
        photpars['av']   = Av
        photpars['rv']   = Rv

        # create filter list and arrange photometry to this list
        sed = self.fppsed.sed(**photpars)

        outdict = {ff_i:sed_i for sed_i,ff_i in zip(sed,self.filterarray)}

        return outdict
