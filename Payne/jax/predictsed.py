#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as np
from jax import lax
from .highred import highAv
import glob

class PayneSEDPredict(object):

    def __init__(self, usebands=None, nnpath=None, singlefile=None, nntype='MLP_v1', norm=True):

        self.nnpath = nnpath
        self.nntype = nntype
        self.norm = norm
        self.singlefile = singlefile

        if usebands == None:
            # user doesn't know which filters, so read in all that
            # are contained in photNN path
            if self.singlefile is not None:
                flist = glob.glob(self.singlefile)
            else:
                flist = glob.glob(self.nnpath+f'/cwc*{self.nntype}*.h5')
            allfilters = []
            for x in flist:
                rootfilename = x.split('/')[-1]
                f = rootfilename.replace('.h5','')
                ss = '_'.join(f.split('_')[4:]) # always going to have [grid,version,nntype,version,filtername]
                allfilters.append(ss)
            usebands = allfilters

        elif isinstance(usebands, str):
            usebands = [usebands]

        else:
            usebands = usebands

        self.anns = self._initphotnn(usebands,nnpath=nnpath)


    def _initphotnn(self, usebands=None, nnpath=None):
        from .photANN_new import modpred

        self.filternames = usebands        

        ANNdict = {}
        for ff in self.filternames:
            try:
                if self.singlefile is not None:
                    nnfile = self.singlefile
                else:
                    nnfile = glob.glob(nnpath + f'/cwc_*{ff}.h5')[0]
                ANNdict[ff] = modpred(nnfile, nntype=self.nntype, norm=self.norm)
            except:
                print(f'Cannot find NN HDF5 file for {nnpath + f"/cwc_*{ff}.h5"}')
        return ANNdict

    def sed(self, logt=None, logg=None, feh=None, afe=None,
            logl=None, av=None, rv=None,
            dist=None, logA=None):
        """
        """

        if type(self.filternames) == type(None):
            filters = self.anns.keys()
            
        inpars = [10.0**logt,logg,feh,afe,av,rv]

        BC = {}
        for f in self.filternames:
            bcpred = self.anns[f].getbc(inpars)
            BC.update(bcpred)

        m = {ff:None for ff in BC.keys()}
        if (logl is not None) and (dist is not None):
            mu = 5.0 * np.log10(dist) - 5.0
            m = {kk: -2.5 * logl + 4.74 - BC[kk] + mu for kk in BC.keys()}
        elif logA is not None:
            m = {kk: 5.0*logA - 10.0*(logt - np.log10(5770.0)) - 0.26 - BC[kk] for kk in BC.keys()}
        else:
            raise IOError('cannot understand input pars into sed function')
        return m

class FastPayneSEDPredict(object):
    
    def __init__(self, usebands=None, nnpath=None):
        from .photANN import ANN, fastANN
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
        inpars = np.asarray([10.0**logt,logg,feh,afe,av,rv])

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