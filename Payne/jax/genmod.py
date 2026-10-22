# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
from jax import lax
import jax.numpy as np
from datetime import datetime

from .fitutils import polycalc

# log10 (R_sun / pc)^2, converts a surface flux F_lambda into the observed
# flux f_lambda = F_lambda * (R/dist)^2 for R in R_sun and dist in pc
LOG_RSUN_PC_SQ = 2.0 * np.log10(6.957E+10 / 3.0856775814913673E+18)

def _avrv_spec_layout(nntype):
    """True for spectral NNs whose genspec pars carry [..., Av, Rv, ...]
    (MLP_v* and the FlxEm family: 'FlxEm', 'FlxEm_v<N>', alias 'FluxEmulator')."""
    return (nntype in ('MLP_v0', 'MLP_v1', 'MLP_v2', 'FluxEmulator')
            or (isinstance(nntype, str) and nntype.startswith('FlxEm')))

class GenMod(object):
    """docstring for GenMod"""
    def __init__(self, *arg, **kwargs):
        super(GenMod, self).__init__()
        self.verbose = kwargs.get('verbose',False)
        
    def _initspecnn(self,nnpath=None,**kwargs):
        from .fitutils import polycalc
        
        self.sNNtype = kwargs.get('nntype',kwargs.get('NNtype',None))
        if self.sNNtype is None:
            self.sNNtype = 'LinNet'

        if self.sNNtype == 'YST1' or self.sNNtype == 'LinNet':
            # if NNtype == 'PC':
            #     from Payne.predict.predictspec_multi import PayneSpecPredict
            # else:
            #     from Payne.predict.ystpred import PayneSpecPredict

            # carbon_bool = kwargs.get('carbon_bool',False)

            from .predictspec import PayneSpecPredict
            # initialize the Payne Spectrum Predictor
            Cnnpath = kwargs.get('Cnnpath',None)
            if Cnnpath is None:
                self.PP = PayneSpecPredict(nnpath=nnpath,NNtype=self.sNNtype)
            else:
                self.PP = PayneSpecPredict(nnpath=nnpath,Cnnpath=Cnnpath,NNtype=self.sNNtype)
        elif _avrv_spec_layout(self.sNNtype):
            from .predictspec import PayneSpecPredictNew
            self.PP = PayneSpecPredictNew(nnpath=nnpath,nntype=self.sNNtype)
        else:
            raise IOError('NNtype not recognized')

    def _initphotnn(self, filterarray, nnpath=None, **kwargs):
        self.filterarray = filterarray
        self.pNNtype = kwargs.get('nntype', kwargs.get('NNtype', "LinNet"))

        if (self.pNNtype is None) or (self.pNNtype == 'LinNet'):
            from .predictsed import FastPayneSEDPredict
            self.fppsed = FastPayneSEDPredict(
                usebands=self.filterarray, nnpath=nnpath,
            )
        elif self.pNNtype in ('MLP_v0', 'MLP_v1', 'MLP_v2'):
            self.cwcversion = kwargs.get('cwcversion','v1.0')
            self.nnversion = kwargs.get('nnversion',None)
            from .predictsed import PayneSEDPredict
            self.fppsed = PayneSEDPredict(
                usesys=self.filterarray, nnpath=nnpath, nntype=self.pNNtype,
                cwcversion=self.cwcversion, nnversion=self.nnversion,
                norm=kwargs.get('norm', True),
                singlefile=kwargs.get('singlefile', None),
            )
        else:
            raise IOError('NNtype not recognized')

        print(f"[ThePayne] Initialized photometric NN with:")
        print(f"[ThePayne] type: {self.pNNtype}")
        print(f"[ThePayne] filters {self.filterarray}")
        print(f"[ThePayne] files: {self.fppsed.flist}")

        if self.filterarray is None:
            self.filterarray = self.fppsed.filternames
            
    def genspec(self,pars,outwave=None,verbose=False,modpoly=False,
                normalize=False,logR=None,dist=None):
        """
        Predict a spectrum on outwave.

        pars : [Teff, logg, FeH, aFe, vrad, vrot, vmic, inst_R, pc0, pc1, ...]
            or, for NNs with Av/Rv inputs (MLP_v*, FlxEm*),
            [Teff, logg, FeH, aFe, vrad, vrot, vmic, inst_R, Av, Rv, pc0, pc1, ...]

        For FlxEm NNs the model is F_lambda in erg/s/cm^2/A, reddened by Av, Rv.
        By default this is the surface flux of the star.

        modpoly   : multiply the model by the Chebyshev polynomial pc0, pc1, ...
        normalize : divide the model by its mean before applying the polynomial
        logR, dist: if given (log10 R in R_sun, dist in pc), scale the model to
                    the observed flux, F_lambda * (R/dist)^2
        """
        # define parameters from pars array
        Teff = pars[0]
        logg = pars[1]
        FeH  = pars[2]
        aFe = pars[3]
        
        indict = {}
        indict['teff']   = Teff
        indict['logg']   = logg
        indict['feh']    = FeH
        indict['afe']    = aFe
        
        if _avrv_spec_layout(self.sNNtype):
            radvel = pars[4]
            rotvel = pars[5]
            vmic = pars[6]
            inst_R = pars[7]
            av = pars[8]
            rv = pars[9]
            pcindstart = 10

            indict['av']     = av
            indict['rv']     = rv

        else:
            radvel = pars[4]
            rotvel = pars[5]
            vmic = pars[6]
            inst_R = pars[7]
            pcindstart = 8

        indict['rad_vel'] = radvel
        indict['rot_vel'] = rotvel
        indict['vmic']   = vmic
        indict['inst_R'] = inst_R
        indict['outwave'] = outwave
        indict['verbose'] = False

        # if verbose:
        #     jax.debug.print('Teff   = {}'.format(Teff))
        #     jax.debug.print('log(g) = {}'.format(logg))
        #     jax.debug.print('[Fe/H] = {}'.format(FeH))
        #     jax.debug.print('[a/Fe] = {}'.format(aFe))
        #     jax.debug.print('Vrad   = {}'.format(radvel))
        #     jax.debug.print('Vstar  = {}'.format(rotvel))
        #     jax.debug.print('Vmic   = {}'.format(vmic))
        #     # print('InstR  = {}'.format(inst_R))
        #     if self.sNNtype in ('MLP_v0', 'MLP_v1', 'MLP_v2'):
        #         jax.debug.print('Av     = {}'.format(av))
        #         jax.debug.print('Rv     = {}'.format(rv))        

        # predict model flux at model wavelengths
        modwave_i,modflux_i = self.PP.getspec(**indict)
        # modwave_i,modflux_i = self.PP.getspec(
        #     Teff=Teff,logg=logg,feh=FeH,afe=aFe,rad_vel=radvel,rot_vel=rotvel,
        #     inst_R=inst_R,vmic=vmic,
        #     outwave=outwave, verbose=True)       

        def modpolyfn(wave):            
            polycoef = pars[pcindstart:]
            polycoef += [0]
            epoly = polycalc(polycoef,wave)
            return epoly

        def modpolydefault(wave):
            return np.ones(len(wave))

        # divide the model by its mean, before the polynomial so pc0 sets the scale
        modflux_i = modflux_i / np.where(normalize, np.mean(modflux_i), 1.0)

        # scale the surface flux to the observed flux using the radius and distance
        if (logR is None) != (dist is None):
            raise ValueError('genspec needs both logR and dist to scale the flux')
        if logR is not None:
            if not _avrv_spec_layout(self.sNNtype):
                raise ValueError(f'logR and dist need a NN that predicts F_lambda, not {self.sNNtype}')
            modflux_i = modflux_i * 10.0**(LOG_RSUN_PC_SQ + 2.0*logR - 2.0*np.log10(dist))

        # if polynomial normalization is turned on then multiply model by it
        epoly = lax.cond(modpoly,modpolyfn,modpolydefault,modwave_i)
        
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

        Rv = lax.cond(rvfree,lambda _:pars[7],lambda _ :3.1,None)

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

        # old NN return list of BCs, newer models return dictionaries
        if isinstance(sed,dict):
            outdict = sed
        else:
            outdict = {ff_i:sed_i for sed_i,ff_i in zip(sed,self.filterarray)}

        return outdict

    def genphot_scaled(self,pars,verbose=False):
        # define parameters from pars array
        Teff = pars[0]
        logg = pars[1]
        FeH  = pars[2]
        aFe  = pars[3]
        logA = pars[4]
        Av   = pars[5]
        # Rv   = pars[6]

        logTeff = np.log10(Teff)

        # create parameter dictionary
        photpars = {}
        photpars['logt'] = logTeff
        photpars['logg'] = logg
        photpars['feh']  = FeH
        photpars['afe']  = aFe
        photpars['logA'] = logA
        photpars['av']   = Av
        photpars['rv']   = 3.1 #Rv

        # create filter list and arrange photometry to this list
        sed = self.fppsed.sed(**photpars)

        outdict = {ff_i:sed_i for sed_i,ff_i in zip(sed,self.filterarray)}

        return outdict
