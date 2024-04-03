# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys,glob,warnings
import numpy as np
from numpy.lib import recfunctions as rfn
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import h5py
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import beta
from datetime import datetime

import Payne
from .smoothing import smoothspec

class readc3k(object):
    def __init__(self,**kwargs):
        # define aliases for the MIST isochrones and C3K/CKC files
        self.MISTpath = kwargs.get('MISTpath',Payne.__abspath__+'data/MIST/MIST_1.2_EEPtrk.h5')
        self.C3Kpath  = kwargs.get('C3Kpath',Payne.__abspath__+'data/C3K/')

        if self.MISTpath is None:
            self.MISTpath = Payne.__abspath__+'data/MIST/MIST_1.2_EEPtrk.h5'

        if self.C3Kpath is None:
            self.C3Kpath = Payne.__abspath__+'data/C3K/'

        # load MIST models
        self.MIST = h5py.File(self.MISTpath,'r')
        self.MISTindex = list(self.MIST['index'])
        # convert btye strings to python strings
        self.MISTindex = [x.decode("utf-8") for x in self.MISTindex]

        # determine the FeH and aFe arrays for C3K
        self.FeHarr = []
        self.alphaarr = []
        self.vtarr = []
        for indinf in glob.glob(self.C3Kpath+'c3k*h5'):
            feh_i =  float(indinf.partition('feh')[-1][:5])
            afe_i =  float(indinf.partition('afe')[-1][:4])
            self.FeHarr.append(feh_i)
            self.alphaarr.append(afe_i)
            if 'vt' in indinf:
                vt_i = float(indinf.partition('vt')[-1][:3])/10.0
                self.vtarr.append(vt_i)
        self.FeHarr = np.unique(self.FeHarr)
        self.alphaarr = np.unique(self.alphaarr)

        vtfixbool = kwargs.get('vtfixed',False)
        if vtfixbool:
            if len(self.vtarr) > 0:
                self.vtarr = [1.0]
        else:
            self.vtarr = np.unique(self.vtarr)

        self.verbose = kwargs.get('verbose',False)

        if self.verbose:
            print('FOUND {} FeH'.format(len(self.FeHarr)))
            print('FOUND {} aFe'.format(len(self.alphaarr)))
            print('FOUND {} Vt'.format(len(self.vtarr)))

        # remove the super metal-rich models that only have aFe = 0
        # if 0.75 in self.FeHarr:
        # 	self.FeHarr.remove(0.75)
        # if 1.00 in self.FeHarr:
        # 	self.FeHarr.remove(1.00)
        # if 1.25 in self.FeHarr:
        # 	self.FeHarr.remove(1.25)
        self.FeHarr = self.FeHarr[self.FeHarr <= 0.5]

        # determine the MIST FeH and aFe arrays
        self.MISTFeHarr = []
        self.MISTalphaarr = []
        for indinf in self.MISTindex:
            self.MISTFeHarr.append(float(indinf.split('/')[0]))
            self.MISTalphaarr.append(float(indinf.split('/')[1]))

        # create weights for Teff
        # determine the min/max Teff from MIST
        self.MISTTeffmin = np.inf
        self.MISTTeffmax = -np.inf
        for ind in self.MISTindex:
            MISTTeffmin_i = self.MIST[ind]['log_Teff'].min()
            MISTTeffmax_i = self.MIST[ind]['log_Teff'].max()
            if MISTTeffmin_i < self.MISTTeffmin:
                self.MISTTeffmin = MISTTeffmin_i
            if MISTTeffmax_i > self.MISTTeffmax:
                self.MISTTeffmax = MISTTeffmax_i

        self.teffwgts = {}
        for ind in self.MISTindex:
            self.teffwgts[ind] = beta(0.2,1.5,
                loc=self.MISTTeffmin-0.1,
                scale=(self.MISTTeffmax+0.1)-(self.MISTTeffmin-0.1)
                ).pdf(self.MIST[ind]['log_Teff'])
            self.teffwgts[ind] = self.teffwgts[ind]/np.sum(self.teffwgts[ind])
        # self.teffwgts = beta(0.5,1.0,loc=self.MISTTeffmin-0.1,scale=(self.MISTTeffmax+0.1)-(self.MISTTeffmin-0.1))

        # create weights for [Fe/H]
        self.fehwgts = beta(1.0,1.0,loc=-4.1,scale=4.7).pdf(self.FeHarr)
        self.fehwgts = self.fehwgts/np.sum(self.fehwgts)
            
        # create a dictionary for the C3K models and populate it for different
        # metallicities
        self.C3K = {}
        for aa in self.alphaarr:
            self.C3K[aa] = {}
            for mm in self.FeHarr:
                if len(self.vtarr) == 0:
                    # glob file name to see if feh/afe file is in c3kpath
                    fnamelist = glob.glob(self.C3Kpath+'c3k*feh{0:+4.2f}_afe{1:+3.1f}.*.h5'.format(mm,aa))
                    if len(fnamelist) == 1:
                        fname = fnamelist[0]
                    else:
                        raise IOError('Could not find suitable C3K file: FeH={0}. aFe={1}'.format(mm,aa))
                    self.C3K[aa][mm] = h5py.File(
                        fname,
                        'r', libver='latest', swmr=True)
                else:
                    self.C3K[aa][mm] = {}
                    for vv in self.vtarr:
                        # glob file name to see if feh/afe file is in c3kpath
                        fnamelist = glob.glob(self.C3Kpath+'c3k*feh{0:+4.2f}*afe{1:+3.1f}*vt{2:02.0f}*h5'.format(mm,aa,vv*10))
                        if len(fnamelist) == 1:
                            fname = fnamelist[0]
                        else:
                            raise IOError('Could not find suitable C3K file: FeH={0}. aFe={1} vt={2}'.format(mm,aa,vv))
                        print(fname)
                        self.C3K[aa][mm][vv] = h5py.File(
                            fname,
                            'r', libver='latest', swmr=True)
                        # # add vtrub to parameter labels
                        # pars_i = self.C3K[aa][mm][vv]['parameters'][:]
                        # pars = np.lib.recfunctions.rec_append_fields(pars_i,'vt',x,dtypes=float)
                        # self.C3K[aa][mm][vv]['parameters'] = pars

        # create min-max dictionary for input labels
        if len(self.vtarr) == 0:
            self.minmax = ({
                'teff': [2500.0,10000.0],
                'logg': [-1,5.5],
                'feh':  [-4.0,0.5],
                'afe':  [-0.2,0.6],
                })
        else:
            self.minmax = ({
                'teff': [2500.0,10000.0],
                'logg': [-1,5.5],
                'feh':  [-4.0,0.5],
                'afe':  [-0.2,0.6],
                'vturb':[0.5,3.0],
                })


        # create min-max for spectra
        self.Fminmax = [0.0,1.0]


    def pullspectra(self,num,**kwargs):
        '''
        Randomly draw num spectra from C3K with option to 
        base draw on the MIST isochrones.
        
        :params num:
            Number of spectra randomly drawn 

        :params label (optional):
            kwarg defined as labelname=[min, max]
            This constrains the spectra to only 
            be drawn from a given range of labels

        :params excludelabels (optional):
            kwarg defined as array of labels
            that should not be included in 
            output sample of spectra. Useful 
            for when defining validation and 
            testing spectra

        : params waverange (optional):
            kwarg used to set wavelength range
            of output spectra
        
        : params reclabelsel (optional):
            kwarg boolean that returns arrays that 
            give how the labels were selected

        : params returncontinuua (optional):
            kwarg boolean that returns contiunuua in 
            addition to the normalized spectra

        : returns spectra:
            Structured array: wave, spectra1, spectra2, spectra3, ...
            where spectrai is a flux array for ith spectrum. wave is the
            wavelength array in nm.

        : returns labels:
            Array of labels for the individual drawn spectra

        : returns wavelenths:
            Array of wavelengths for predicted spectra

        '''

        Teffrange = kwargs.get('Teff',None)
        if Teffrange is None:
            Teffrange = [2500.0,15000.0]

        loggrange = kwargs.get('logg',None)
        if loggrange is None:
            loggrange = [-1.0,5.0]

        fehrange = kwargs.get('FeH',None)
        if fehrange is None:
            fehrange = [-4.0,0.5]

        aFerange = kwargs.get('aFe',None)
        if aFerange is None:
            aFerange = [-0.2,0.6]

        vtrange = kwargs.get('vturb',None)
        if vtrange is None:
            vtrange = [0.5,3.0]

        if 'resolution' in kwargs:
            resolution = kwargs['resolution']
        else:
            resolution = None

        if 'excludelabels' in kwargs:
            excludelabels = kwargs['excludelabels'].T.tolist()
        else:
            excludelabels = []

        # default is just the MgB triplet 
        waverange = kwargs.get('waverange',[5150.0,5300.0])

        # set up some booleans
        dividecont    = kwargs.get('dividecont',True)
        reclabelsel   = kwargs.get('reclabelsel',False)
        continuuabool = kwargs.get('returncontinuua',False)
        MISTweighting = kwargs.get('MISTweighting',False)
        # timeit        = kwargs.get('timeit',False)

        # randomly select num number of MIST isochrone grid points, currently only 
        # using dwarfs, subgiants, and giants (EEP = 200-808)

        labels = []
        spectra = []
        wavelength_o_flag = True
        if reclabelsel:
            initlabels = []
        if continuuabool:
            continuua = []

        for ii in range(num):
            if self.verbose:
                print(f'... {ii+1}')
                starttime = datetime.now()

            while True:
                # first randomly draw a [Fe/H]
                while True:
                    if MISTweighting:
                        p_i = self.fehwgts
                    else:
                        p_i = None
                    FeH_i = np.random.choice(self.FeHarr,p=p_i)

                    # check to make sure FeH_i is in user defined 
                    # [Fe/H] limits
                    if (FeH_i >= fehrange[0]) & (FeH_i <= fehrange[1]):
                        break
                # if self.verbose:
                # 	print('Pulled random [Fe/H] in {0}'.format(datetime.now()-starttime))

                # then draw an alpha abundance
                while True:
                    alpha_i = np.random.choice(self.alphaarr)

                    # check to make sure alpha_i is in user defined
                    # [alpha/Fe] limits
                    if (alpha_i >= aFerange[0]) & (alpha_i <= aFerange[1]):
                        break
                # if self.verbose:
                # 	print('Pulled random [a/Fe] in {0}'.format(datetime.now()-starttime))

                if len(self.vtarr) > 0:
                    # then draw an vturb
                    while True:
                        vt_i = np.random.choice(self.vtarr)

                        # check to make sure vt_i is in user defined
                        # vturb limits
                        if (vt_i >= vtrange[0]) & (vt_i <= vtrange[1]):
                            break
                    # if self.verbose:
                    # 	print('Pulled random vturb in {0}'.format(datetime.now()-starttime))				

                if len(self.vtarr) > 0:
                    # select the C3K spectra at that [Fe/H], [alpha/Fe], vturb
                    C3K_i = self.C3K[alpha_i][FeH_i][vt_i]
                    # create array of all labels in specific C3K file
                    C3Kpars = np.array(C3K_i['parameters'])
                    # tack on vturb to parameter array
                    C3Kpars = rfn.rec_append_fields(
                        C3Kpars,'vt',
                        vt_i*np.ones(C3Kpars.shape[0],dtype=float),
                        dtypes=float)
                else:
                    # select the C3K spectra at that [Fe/H] and [alpha/Fe]
                    C3K_i = self.C3K[alpha_i][FeH_i]
                    # create array of all labels in specific C3K file
                    C3Kpars = np.array(C3K_i['parameters'])

                # convert log(teff) to teff
                C3Kpars['logt'] = 10.0**C3Kpars['logt']
                C3Kpars = rfn.rename_fields(C3Kpars,{'logt':'teff'})

                # if self.verbose:
                # 	print('create arrray of C3Kpars in {0}'.format(datetime.now()-starttime))

                # select the range of MIST models with that [Fe/H]
                # first determine the FeH and aFe that are nearest to MIST values
                FeH_i_MIST = self.MISTFeHarr[np.argmin(np.abs(FeH_i-self.MISTFeHarr))]
                aFe_i_MIST = self.MISTalphaarr[np.argmin(np.abs(alpha_i-self.MISTalphaarr))]
                MIST_i = self.MIST['{0:4.2f}/{1:4.2f}/0.40'.format(FeH_i_MIST,aFe_i_MIST)]

                # if self.verbose:
                # 	print('Pulled MIST models in {0}'.format(datetime.now()-starttime))

                if MISTweighting:
                    # # generate Teff weights
                    # teffwgts_i = self.teffwgts.pdf(MIST_i['log_Teff'])
                    # teffwgts_i = teffwgts_i/np.sum(teffwgts_i)
                    teffwgts_i = self.teffwgts['{0:4.2f}/{1:4.2f}/0.40'.format(FeH_i_MIST,aFe_i_MIST)]
                else:
                    teffwgts_i = None

                # if self.verbose:
                # 	print('Created MIST weighting {0}'.format(datetime.now()-starttime))

                while True:
                    # randomly select a EEP, log(age) combination with weighting 
                    # towards the hotter temps if user wants
                    MISTsel = np.random.choice(len(MIST_i),p=teffwgts_i)

                    # get MIST Teff and log(g) for this selection
                    logt_MIST_i,logg_MIST_i = MIST_i[MISTsel]['log_Teff'], MIST_i[MISTsel]['log_g']

                    # check to make sure MIST log(g) and log(Teff) have a spectrum in the C3K grid
                    # if not draw again
                    if (
                        (logt_MIST_i >= np.log10(Teffrange[0])) and (logt_MIST_i <= np.log10(Teffrange[1])) and
                        (logg_MIST_i >= loggrange[0]) and (logg_MIST_i <= loggrange[1])
                        ):
                        break
                # if self.verbose:
                # 	print('Selected MIST pars in {0}'.format(datetime.now()-starttime))


                # add a gaussian blur to the MIST selected Teff and log(g)
                # sigma_t = 750K, sigma_g = 1.5
                randomT = np.random.randn()*500.0
                randomg = np.random.randn()*0.5

                # check to see if randomT is an issue for log10
                if 10.0**logt_MIST_i + randomT <= 0.0:
                    randomT = np.abs(randomT)
                    
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        logt_MIST = np.log10(10.0**logt_MIST_i + randomT)				
                        logg_MIST = logg_MIST_i + randomg
                    except Warning:
                        print(
                            'Caught a MIST parameter that does not make sense: {0} {1} {2} {3}'.format(
                                randomT,10.0**logt_MIST_i,randomg,logg_MIST_i))
                        logt_MIST = logt_MIST_i
                        logg_MIST = logg_MIST_i

                # do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
                C3KNN = NearestNDInterpolator(
                    np.array([C3Kpars['teff'],C3Kpars['logg']]).T,range(0,len(C3Kpars))
                    )((10.0**logt_MIST,logg_MIST))
                C3KNN = int(C3KNN)


                # determine the labels for the selected C3K spectrum
                try:
                    label_i = list(C3Kpars[C3KNN])
                except IndexError:
                    print(C3KNN)
                    raise

                # if self.verbose:
                # 	print('Determine C3K labels in {0}'.format(datetime.now()-starttime))

                # check to see if user defined labels to exclude, if so
                # continue on to the next iteration
                if label_i in excludelabels:
                    print('Found spectrum in exclude labels')
                    continue

                # turn off warnings for this step, C3K has some continuaa with flux = 0
                if dividecont:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        spectra_i = C3K_i['spectra'][C3KNN]/C3K_i['continuua'][C3KNN]
                else:
                    spectra_i = C3K_i['spectra'][C3KNN]/np.nanmedian(C3K_i['spectra'][C3KNN])

                if continuuabool:
                    continuua_i = C3K_i['continuua'][C3KNN]
                else:
                    continuua_i = None

                # if self.verbose:
                    # print('Create C3K spectra in {0}'.format(datetime.now()-starttime))

                # check to see if label_i in labels
                # if so, then skip the append and go to next step in while loop
                # do this before the smoothing to reduce run time
                if (label_i in labels):
                    continue

                # check to see if spectrum has nan's, if so remove them as 
                # long as they are < 0.1% of the total number of pixels
                if (np.isfinite(spectra_i).sum() != len(spectra_i)):
                    print(f'Found {np.isnan(spectra_i).sum()} NaN out of {len(spectra_i)}')
                    print(label_i)
                    continue

                # store a wavelength array as an instance, all of C3K has 
                # the same wavelength sampling
                if wavelength_o_flag:
                    wavelength_o = [] # initialize the output wavelength array
                    wavelength_o_flag = False # turn off this step for all subsequent models
                    wavelength_i = np.array(C3K_i['wavelengths'])
                    if resolution != None:
                        # define new wavelength array with 3*resolution element sampling
                        i = 1
                        while True:
                            wave_i = waverange[0]*(1.0 + 1.0/(3.0*resolution))**(i-1.0)
                            if wave_i <= waverange[1]:
                                wavelength_o.append(wave_i)
                                i += 1
                            else:
                                break
                        wavelength_o = np.array(wavelength_o)
                    else:
                        wavecond = (wavelength_i >= waverange[0]) & (wavelength_i <= waverange[1])
                        wavecond = np.array(wavecond,dtype=bool)
                        wavelength_o = wavelength_i[wavecond]

                # if self.verbose:
                # 	print('Saved a C3K wavelength instance in {0}'.format(datetime.now()-starttime))

                # if user defined resolution to train at, the smooth C3K to that resolution
                if resolution != None:
                    spectra_i = self.smoothspecfunc(wavelength_i,spectra_i,resolution,
                        outwave=wavelength_o,smoothtype='R',fftsmooth=True)
                else:
                    spectra_i = spectra_i[wavecond]

                if continuuabool:
                    if resolution != None:
                        continuua_i = self.smoothspecfunc(wavelength_i,continuua_i,resolution,
                            outwave=wavelength_o,smoothtype='R',fftsmooth=True)
                    else:
                        continuua_i = continuua_i[wavecond]


                # if self.verbose:
                # 	print('Convolve C3K to new R in {0}'.format(datetime.now()-starttime))

                labels.append(label_i)
                spectra.append(spectra_i)

                # if requested, return continuua
                if continuuabool:
                    continuua.append(continuua_i)

                # if requested, record random selected parameters
                if reclabelsel:
                    if len(self.vtarr) > 0:
                        initlabels.append([10.0**logt_MIST,logg_MIST,FeH_i,alpha_i,vt_i])
                    else:
                        initlabels.append([10.0**logt_MIST,logg_MIST,FeH_i,alpha_i])
                break
            if self.verbose:
                print(f'-> Added {ii+1}, total time: {0}'.format(datetime.now()-starttime))

        output = [np.array(spectra), np.array(labels),wavelength_o]
        if reclabelsel:
            output += [np.array(initlabels)]
        if continuuabool:
            output += [np.array(continuua)]

        return output


    def selspectra(self,inlabels,**kwargs):
        '''
        specifically select and return C3K spectra at user
        defined labels

        :param inlabels
        Array of user defined lables for returned C3K spectra
        format is [Teff,logg,FeH,aFe]

        '''

        if 'resolution' in kwargs:
            resolution = kwargs['resolution']
        else:
            resolution = None

        if 'waverange' in kwargs:
            waverange = kwargs['waverange']
        else:
            # default is just the MgB triplet 
            waverange = [5150.0,5200.0]

        # user wants to return continuua
        continuuabool = kwargs.get('returncontinuua',False)
        dividecont    = kwargs.get('dividecont',True)

        labels = []
        spectra = []
        wavelength_o_flag = True

        if continuuabool:
            continuua = []

        if isinstance(inlabels[0],float):
            inlabels = [inlabels]

        for li in inlabels:
            # select the C3K spectra at that [Fe/H] and [alpha/Fe]
            teff_i  = li[0]
            logg_i  = li[1]
            FeH_i   = li[2]
            alpha_i = li[3]

            if len(self.vtarr) > 0:
                vt_i = li[4]

            # find nearest value to FeH and aFe
            try:
                FeH_i   = self.FeHarr[np.argmin(np.abs(np.array(self.FeHarr)-FeH_i))]
            except:
                print('Issue with finding nearest FeH')
                print(self.FeHarr)
                print(FeH_i)
                raise

            try:
                alpha_i = self.alphaarr[np.argmin(np.abs(np.array(self.alphaarr)-alpha_i))]
            except:
                print('Issue with finding nearest aFe')
                print(self.alphaarr)
                print(alpha_i)
                raise				

            if len(self.vtarr) > 0:
                # select the C3K spectra at that [Fe/H], [alpha/Fe], vturb
                C3K_i = self.C3K[alpha_i][FeH_i][vt_i]
                # create array of all labels in specific C3K file
                C3Kpars = np.array(C3K_i['parameters'])
                # tack on vturb to parameter array
                C3Kpars = rfn.rec_append_fields(
                    C3Kpars,'vt',
                    vt_i*np.ones(C3Kpars.shape[0],dtype=float),
                    dtypes=float)
            else:
                # select the C3K spectra at that [Fe/H] and [alpha/Fe]
                C3K_i = self.C3K[alpha_i][FeH_i]
                # create array of all labels in specific C3K file
                C3Kpars = np.array(C3K_i['parameters'])

            # convert log(teff) to teff
            C3Kpars['logt'] = 10.0**C3Kpars['logt']
            C3Kpars = rfn.rename_fields(C3Kpars,{'logt':'teff'})

            # do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
            C3KNN = NearestNDInterpolator(
                np.array([C3Kpars['teff'],C3Kpars['logg']]).T,range(0,len(C3Kpars))
                )((teff_i,logg_i))
            C3KNN = int(C3KNN)

            # determine the labels for the selected C3K spectrum
            try:
                label_i = list(C3Kpars[C3KNN])
            except IndexError:
                print(C3KNN)
                raise

            # turn off warnings for this step, C3K has some continuaa with flux = 0
            if dividecont:
                with np.errstate(divide='ignore', invalid='ignore'):
                    spectra_i = C3K_i['spectra'][C3KNN]/C3K_i['continuua'][C3KNN]
            else:
                spectra_i = C3K_i['spectra'][C3KNN]/np.nanmedian(C3K_i['spectra'][C3KNN])
                
            if continuuabool:
                continuua_i = C3K_i['continuua'][C3KNN]
            else:
                continuua_i = None

            # check to see if label_i in labels, or spectra_i is nan's
            # if so, then skip the append and go to next step in while loop
            # do this before the smoothing to reduce run time
            # if np.any(np.isnan(spectra_i)):
            # 	continue

            # store a wavelength array as an instance, all of C3K has 
            # the same wavelength sampling
            if wavelength_o_flag:
                wavelength_o = [] # initialize the output wavelength array
                wavelength_o_flag = False # turn off this step for all subsequent models
                wavelength_i = np.array(C3K_i['wavelengths'])
                if resolution != None:
                    # define new wavelength array with 3*resolution element sampling
                    i = 1
                    while True:
                        wave_i = waverange[0]*(1.0 + 1.0/(3.0*resolution))**(i-1.0)
                        if wave_i <= waverange[1]:
                            wavelength_o.append(wave_i)
                            i += 1
                        else:
                            break
                    wavelength_o = np.array(wavelength_o)
                else:
                    wavecond = (wavelength_i >= waverange[0]) & (wavelength_i <= waverange[1])
                    wavecond = np.array(wavecond,dtype=bool)
                    wavelength_o = wavelength_i[wavecond]

            # if user defined resolution to train at, the smooth C3K to that resolution
            if resolution != None:
                spectra_i = self.smoothspecfunc(wavelength_i,spectra_i,resolution,
                    outwave=wavelength_o,smoothtype='R',fftsmooth=True)
            else:
                spectra_i = spectra_i[wavecond]

            if continuuabool:
                if resolution != None:
                    continuua_i = self.smoothspecfunc(wavelength_i,continuua_i,resolution,
                        outwave=wavelength_o,smoothtype='R',fftsmooth=True)
                else:
                    continuua_i = continuua_i[wavecond]

            labels.append(label_i)
            spectra.append(spectra_i)
            if continuuabool:
                continuua.append(continuua_i)

        output = [np.array(spectra), np.array(labels), wavelength_o]

        if continuuabool:
            output += [np.array(continuua)]

        return output

    def pullpixel(self,pixelnum,**kwargs):
        # a convience function if you only want to pull one pixel at a time
        # it also does a check and remove any NaNs in spectra 


        Teffrange = kwargs.get('Teff',None)
        if Teffrange == None:
            Teffrange = [2500.0,15000.0]

        loggrange = kwargs.get('logg',None)
        if loggrange == None:
            loggrange = [-1.0,5.0]

        fehrange = kwargs.get('FeH',None)
        if fehrange == None:
            fehrange = [min(self.FeHarr),max(self.FeHarr)]

        aFerange = kwargs.get('aFe',None)
        if aFerange == None:
            aFerange = [min(self.alphaarr),max(self.alphaarr)]

        if 'resolution' in kwargs:
            resolution = kwargs['resolution']
        else:
            resolution = None

        if 'excludelabels' in kwargs:
            excludelabels = kwargs['excludelabels']
        else:
            excludelabels = []

        if 'waverange' in kwargs:
            waverange = kwargs['waverange']
        else:
            # default is just the MgB triplet 
            waverange = [5145.0,5300.0]

        if 'reclabelsel' in kwargs:
            reclabelsel = kwargs['reclabelsel']
        else:
            reclabelsel = False

        if 'MISTweighting' in kwargs:
            MISTweighting = kwargs['MISTweighting']
        else:
            MISTweighting = True

        # if 'timeit' in kwargs:
        # 	timeit = kwargs['timeit']
        # else:
        # 	timeit = False

        if 'inlabels' in kwargs:
            inlabels = kwargs['inlabels']
        else:
            inlabels = []

        if 'num' in kwargs:
            num = kwargs['num']
        else:
            num = 1

        if inlabels == []:				
            # pull the spectrum
            spectra,labels,wavelength = self(
                num,resolution=resolution, waverange=waverange,
                MISTweighting=MISTweighting)

        else:
            spectra,labels,wavelength = self.selspectra(
                inlabels,resolution=resolution, waverange=waverange)

        # select individual pixels
        pixelarr = np.array(spectra[:,pixelnum])
        labels = np.array(labels)

        # # determine if an of the pixels are NaNs
        # mask = np.ones_like(pixelarr,dtype=bool)
        # nanval = np.nonzero(np.isnan(pixelarr))
        # numnan = len(nanval[0])
        # mask[np.nonzero(np.isnan(pixelarr))] = False

        # # remove nan pixel values and labels
        # pixelarr = pixelarr[mask]
        # labels = labels[mask]
        
        return pixelarr, labels, wavelength

    def checklabels(self,inlabels,**kwargs):
        # a function that allows the user to determine the nearest C3K labels to array on input labels
        # useful to run before actually selecting spectra

        labels = []

        for li in inlabels:
            # select the C3K spectra at that [Fe/H] and [alpha/Fe]
            teff_i  = li[0]
            logg_i  = li[1]
            FeH_i   = li[2]
            alpha_i = li[3]

            # find nearest value to FeH and aFe
            FeH_i   = self.FeHarr[np.argmin(np.abs(self.FeHarr-FeH_i))]
            alpha_i = self.alphaarr[np.argmin(np.abs(self.alphaarr-alpha_i))]

            # select the C3K spectra for these alpha and FeH
            C3K_i = self.C3K[alpha_i][FeH_i]

            # create array of all labels in specific C3K file
            C3Kpars = np.array(C3K_i['parameters'])

            # do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
            C3KNN = NearestNDInterpolator(
                np.array([C3Kpars['logt'],C3Kpars['logg']]).T,range(0,len(C3Kpars))
                )((teff_i,logg_i))
            C3KNN = int(C3KNN)

            # determine the labels for the selected C3K spectrum
            label_i = list(C3Kpars[C3KNN])		
            labels.append(label_i)

        return np.array(labels)


    def smoothspecfunc(self,wave, spec, sigma, outwave=None, **kwargs):
        outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
        return outspec
