from Payne.fitting import fitstar
from astropy.table import Table
import numpy as np
import sys

print('-------- RUNNING MOCK SOLAR DEMO ---------')

demospec = Table.read('demo_spec.fits',format='fits')

inputdict = {}
inputdict['spec'] = {}
inputdict['specANNpath'] = '/Users/pcargile/Astro/GITREPOS/ThePayne/data/specANN/Hecto_T2K_C3K.h5'
inputdict['spec']['obs_wave'] = demospec['WAVE']
inputdict['spec']['obs_flux'] = demospec['FLUX']
# error of SNR = 50
inputdict['spec']['obs_eflux'] = demospec['FLUX']/50.0
inputdict['spec']['normspec'] = True

inputdict['phot'] = {}
inputdict['photANNpath'] = '/Users/pcargile/Astro/GITREPOS/ThePayne/data/photANN/'
inputdict['phot']['Tycho_B'] = [6.9169,0.1]
inputdict['phot']['Tycho_V'] = [5.8940,0.1]

inputdict['phot']['2MASS_J'] = [3.94311,0.1]
inputdict['phot']['2MASS_H'] = [3.50567,0.1]
inputdict['phot']['2MASS_Ks'] =[3.40414,0.1]

inputdict['sampler'] = {}
inputdict['sampler']['samplemethod'] = 'rwalk'
inputdict['sampler']['npoints'] = 50
inputdict['sampler']['samplertype'] = 'single'
inputdict['sampler']['flushnum'] = 100

inputdict['output'] = 'OUTPUTFILE.dat'

# set some flat priors for defining the prior volume
inputdict['priordict'] = {}
inputdict['priordict']['Teff']   = {'uniform':[5000.0,6500.0]}
inputdict['priordict']['log(g)'] = {'uniform':[3.0,5.0]}
inputdict['priordict']['[Fe/H]'] = {'uniform':[-0.25,0.25]}
inputdict['priordict']['Dist']   = {'uniform':[5.0,20.0]}

inputdict['priordict']['blaze_coeff'] = ([
	[0.0720479608681,0.0525086134796],
	[-0.261792998145,0.0575258483974],
	[0.000243761834872,0.00702220288182],
	[-0.0495597298557,0.043602355668],
	[0.0315056023735,0.035893863888],
	[-0.0127712929987,0.0297117232166],
	[0.0301115761372,0.0351109513734],
	[0.000130673860212,0.0106496054721],
	[0.00425294373254,0.0189812284416]
	])

# set an additional guassian prior on the instrument profile
inputdict['priordict']['Inst_R'] = {'gaussian':[32000.0,1000.0]}

FS = fitstar.FitPayne()
print('---------------')
print('    PHOT:')
for kk in inputdict['phot'].keys():
        print('       {0} = {1} +/- {2}'.format(kk,inputdict['phot'][kk][0],inputdict['phot'][kk][1]))
print('    Median Spec Flux: ')
print('       {0}'.format(np.median(inputdict['spec']['obs_flux'])))
print('    Median Spec Err_Flux:')
print('       {0}'.format(np.median(inputdict['spec']['obs_eflux'])))
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


