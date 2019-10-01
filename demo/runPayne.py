from Payne.fitting import fitstar
from astropy.table import Table
import h5py
import numpy as np
import sys

runspec = True
runphot = True
runmock = False

startorun = 'Sun'
startorun = 'Procyon'

if startorun == 'Sun':
     if runmock:
          print('-------- RUNNING MOCK SOLAR DEMO ---------')
          print('----- Teff = 5770.0, log(g) = 4.44 -------')
          print('-----  [Fe/H] = 0.0, log(L) = 0.0  -------')
          print('-----    Av = 0.5, Dist = 10.0     -------')
     else:
          print('-------- RUNNING OBS SOLAR DEMO ---------')

if startorun == 'Procyon':
     if runmock:
          print('-------- RUNNING MOCK SOLAR DEMO ---------')
          print('----- Teff = 5770.0, log(g) = 4.44 -------')
          print('-----  [Fe/H] = 0.0, log(L) = 0.0  -------')
          print('-----    Av = 0.5, Dist = 10.0     -------')
     else:
          print('-------- RUNNING OBS PROCYON DEMO ---------')


print('  ---- Running Spec: {}'.format(runspec))
print('  ---- Running Phot: {}'.format(runphot))

inputdict = {}

if runspec:
     inputdict['spec'] = {}
     inputdict['specANNpath'] = '/Users/pcargile/Astro/ThePayne/Hecto_FAL_v7.1/trainedANN/FALANN_RVS31_v7.1.h5'

     if runmock:
          with h5py.File('demodata.h5','r') as th5:
               demoflux = np.array(th5['spec/flux'])
               demowave = np.array(th5['spec/wave'])

          inputdict['spec']['obs_wave'] = demowave
          inputdict['spec']['obs_flux'] = demoflux
          # error of SNR = 50
          inputdict['spec']['obs_eflux'] = demoflux/25.0
          inputdict['spec']['normspec'] = False
          inputdict['spec']['convertair'] = False
          # set an additional guassian prior on the instrument profile
          # inputdict['priordict']['Inst_R'] = {'gaussian':[32000.0,1000.0]}
     else:
          if startorun == 'Sun':
               spec = Table.read('Sun_UVES_R32K.fits',format='fits')
          if startorun == 'Procyon':
               spec = Table.read('Procyon_UVES_R32K.fits',format='fits')               

          spec = spec[(spec['wave'] >= 5145) & (spec['wave'] <= 5325)]
          speccond = (spec['flux'] > 0.0) & (spec['e_flux'] > 0.0) & np.isfinite(spec['flux']) & np.isfinite(spec['e_flux'])
          spec = spec[speccond]
          inputdict['spec']['obs_wave'] = spec['wave']
          inputdict['spec']['obs_flux'] = spec['flux']
          inputdict['spec']['obs_eflux'] = spec['e_flux']
          inputdict['spec']['normspec'] = False
          inputdict['spec']['convertair'] = True


if runphot:
     inputdict['phot'] = {}
     inputdict['photANNpath'] = '/Users/pcargile/Astro/GITREPOS/ThePayne/data/photANN/'

     if runmock:
          # MOCK PHOT
          with h5py.File('demodata.h5','r') as th5:
               filterarr = [x.decode('ascii') for x in th5['phot/filters']]
               phot      = np.array(th5['phot/phot'])

     else:
          if startorun == 'Sun':
               # sort of observed phot of Sun
               phot = ([
                    5.03,4.64,4.52,4.51,
                    3.67,3.32,3.27,
                    3.26,3.28
                    ])
               filterarr = ([
                    'PS_g','PS_r','PS_i','PS_z',
                    '2MASS_J','2MASS_H','2MASS_Ks',
                    'WISE_W1','WISE_W2'])

               inputdict['phot'] = {fn:[p_i,0.05] for fn,p_i in zip(filterarr,phot)}


          if startorun == 'Procyon':
               phot = ([
                    0.73,0.33,0.09,-0.14,
                    -0.41,-0.51,-0.60,
                    ])
               filterarr = ([
                    'Bessell_B','Bessell_V','Bessell_R','Bessell_I',
                    '2MASS_J','2MASS_H','2MASS_Ks',
                    ])

               inputdict['phot'] = {fn:[p_i,0.05] for fn,p_i in zip(filterarr,phot)}


# set parameter for sampler
inputdict['sampler'] = {}
inputdict['sampler']['samplertype'] = 'Static'
inputdict['sampler']['samplerbounds'] = 'multi'
inputdict['sampler']['samplemethod'] = 'rwalk'
inputdict['sampler']['npoints'] = 125
inputdict['sampler']['flushnum'] = 1
inputdict['sampler']['delta_logz_final'] = 0.1
inputdict['sampler']['bootstrap'] = 0
inputdict['sampler']['walks'] = 25

# set some flat priors for defining the prior volume
inputdict['priordict'] = {}
inputdict['priordict']['Teff']   = {'pv_uniform':[4000.0,7000.0]}
inputdict['priordict']['log(g)'] = {'pv_uniform':[4.0,5.5]}
inputdict['priordict']['[Fe/H]'] = {'pv_uniform':[-0.1,0.1]}
inputdict['priordict']['[a/Fe]'] = {'pv_uniform':[-0.1,0.1]}
inputdict['priordict']['Vrad']   = {'pv_uniform':[-1.0,1.0]}
inputdict['priordict']['Vrot']   = {'pv_uniform':[0.0,5.0]}


# inputdict['photscale'] = False
# inputdict['priordict']['Dist']   = {'uniform':[1.0,200.0]}
# inputdict['priordict']['log(R)'] = {'uniform':[-0.1,0.1]}
inputdict['photscale'] = True
inputdict['priordict']['log(A)'] = {'pv_uniform':[-3.0,7.0]}
inputdict['priordict']['Av']     = {'pv_uniform':[0.0,1.0]}

# set an additional guassian prior on the instrument profile
inputdict['priordict']['Inst_R'] = (
     {'pv_tgaussian':[30000.0,37000.0,32000.0,1000.0]}
     )

inputdict['output'] = 'demoout.dat'


FS = fitstar.FitPayne()
print('---------------')
if 'phot' in inputdict.keys():
     print('    PHOT:')
     for kk in inputdict['phot'].keys():
             print('       {0} = {1} +/- {2}'.format(kk,inputdict['phot'][kk][0],inputdict['phot'][kk][1]))
if 'spec' in inputdict.keys():
     print('    Median Spec Flux: ')
     print('       {0}'.format(np.median(inputdict['spec']['obs_flux'])))
     print('    Median Spec Err_Flux:')
     print('       {0}'.format(np.median(inputdict['spec']['obs_eflux'])))

if 'priordict' in inputdict.keys():
     print('    PRIORS:')
     for kk in inputdict['priordict'].keys():
          if kk == 'blaze_coeff':
               pass
          else:
               for kk2 in inputdict['priordict'][kk].keys():
                    if kk2 == 'uniform':
                         print('       {0}: min={1} max={2}'.format(kk,inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))
                    if kk2 == 'gaussian':
                         print('       {0}: N({1},{2})'.format(kk,inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))

print('--------------')

sys.stdout.flush()
result = FS.run(inputdict=inputdict)
sys.stdout.flush()
