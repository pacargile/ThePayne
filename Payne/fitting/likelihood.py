import numpy as np
from .genmod import GenMod
from datetime import datetime

class likelihood(object):
     """docstring for likelihood"""
     def __init__(self,fitargs,fitpars,runbools,**kwargs):
          super(likelihood, self).__init__()

          self.verbose = kwargs.get('verbose',True)
          self.fitargs = fitargs

          # split up the boolean flags
          self.spec_bool = runbools[0]
          self.phot_bool = runbools[1]
          self.modpoly_bool = runbools[2]
          self.photscale_bool = runbools[3]
          self.carbon_bool = runbools[4]

          self.fixedpars = self.fitargs['fixedpars']

          # initialize the model generation class
          self.GM = GenMod()

          # initialize the ANN for spec and phot if user defined
          if self.spec_bool:
               self.GM._initspecnn(nnpath=fitargs['specANNpath'],
                    NNtype=self.fitargs['NNtype'],
                    carbon_bool=self.carbon_bool)
          if self.phot_bool:
               self.GM._initphotnn(self.fitargs['obs_phot'].keys(),
                    nnpath=fitargs['photANNpath'])

          # determine the number of dims
          self.ndim = 0
          self.fitpars_i = []
          for pp in fitpars[0]:
               if fitpars[1][pp]:
                    self.fitpars_i.append(pp)
                    self.ndim += 1

     def lnlikefn(self,pars):
          # build the parameter dictionary
          self.parsdict = {pp:vv for pp,vv in zip(self.fitpars_i,pars)} 

          # add fixed parameters to parsdict
          for kk in self.fixedpars.keys():
               self.parsdict[kk] = self.fixedpars[kk]

          if self.spec_bool:
               specpars = ([
                    self.parsdict[pp] 
                    if (pp in self.parsdict.keys()) else np.nan
                    for pp in ['Teff','log(g)','[Fe/H]','[a/Fe]','Vrad','Vrot','Vmic','Inst_R'] 
                    ])
               if self.modpoly_bool:
                    specpars = specpars + [self.parsdict[pp] for pp in self.fitpars_i if 'pc' in pp]
          else:
               specpars = None

          if self.phot_bool:
               photpars = [self.parsdict[pp] for pp in ['Teff','log(g)','[Fe/H]','[a/Fe]']]
               if 'log(A)' in self.fitpars_i:
                    photpars = photpars + [self.parsdict['log(A)']]
               else:
                    photpars = photpars + [self.parsdict['log(R)'],self.parsdict['Dist']]
               photpars = photpars + [self.parsdict['Av']]
               # include Rv if user wants, else set to 3.1
               if 'Rv' in self.fitpars_i:
                    photpars = photpars + [self.parsdict['Rv']]
               else:
                    photpars = photpars + [None]

          else:
               photpars = None

          # calculate likelihood probability
          lnlike_i = self.lnlike(specpars=specpars,photpars=photpars)

          if lnlike_i == np.nan:
               print(pars,lnlike_i)
          return lnlike_i

     def lnlike(self,specpars=None,photpars=None):

          if self.spec_bool:
               # generate model spectrum
               specmod = self.GM.genspec(specpars,
                    outwave=self.fitargs['obs_wave_fit'],
                    modpoly=self.modpoly_bool,
                    carbon_bool=self.carbon_bool)
               modwave_i,modflux_i = specmod

               # calc chi-square for spec
               specchi2 = np.sum( 
                    [((m-o)**2.0)/(s**2.0) for m,o,s in zip(
                         modflux_i,self.fitargs['obs_flux_fit'],self.fitargs['obs_eflux_fit'])])
          else:
               specchi2 = 0.0

          if self.phot_bool:
               # generate model SED
               if self.photscale_bool:
                    sedmod  = self.GM.genphot_scaled(photpars)
               else:
                    sedmod  = self.GM.genphot(photpars)

               # calculate chi-square for SED
               sedchi2 = np.sum(
                    [((sedmod[kk]-self.fitargs['obs_phot'][kk][0])**2.0)/(self.fitargs['obs_phot'][kk][1]**2.0) 
                    for kk in self.fitargs['obs_phot'].keys()]
                    )
          else:
               sedchi2 = 0.0

          # return ln(like) = -0.5 * chi-square
          return -0.5*(specchi2+sedchi2)
