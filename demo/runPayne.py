from Payne.fitting import fitstar
from astropy.table import Table
import numpy as np
import sys

runspec = True
runphot = False
runmock = True

if runmock:
	print('-------- RUNNING MOCK SOLAR DEMO ---------')
	print('----- Teff = 5770.0, log(g) = 4.44 -------')
	print('-----  [Fe/H] = 0.0, log(L) = 0.0  -------')
	print('-----    Av = 0.5, Dist = 10.0     -------')
else:
	print('-------- RUNNING OBS SOLAR DEMO ---------')

print('  ---- Running Spec: {}'.format(runspec))
print('  ---- Running Phot: {}'.format(runphot))

inputdict = {}

if runspec:
	inputdict['spec'] = {}
	# inputdict['specANNpath'] = '/Users/pcargile/Astro/ThePayne/Hecto_C3K_v2/C3K_Hecto_v2.h5'
	inputdict['specANNpath'] = '/Users/pcargile/Astro/ThePayne/Hecto_C3K_v3/trainedANN/C3KANN_RVS31.h5'

	if runmock:
		demospec = Table.read('demo_spec.fits',format='fits')
		inputdict['spec']['obs_wave'] = demospec['WAVE']
		inputdict['spec']['obs_flux'] = demospec['FLUX']
		# error of SNR = 50
		inputdict['spec']['obs_eflux'] = demospec['FLUX']/100.0
		inputdict['spec']['normspec'] = False
		inputdict['spec']['convertair'] = False
		# set an additional guassian prior on the instrument profile
		# inputdict['priordict']['Inst_R'] = {'gaussian':[32000.0,1000.0]}
	else:
		sunspec = Table.read('ATLAS.Sun_47000.txt.gz',format='ascii')
		sunspec = sunspec[(sunspec['waveobs'] >= 514.5) & (sunspec['waveobs'] <= 532.5)]
		sunspec = sunspec[(sunspec['flux'] != 0.0) & (sunspec['err'] != 0.0)]
		inputdict['spec']['obs_wave'] = sunspec['waveobs']*10.0
		inputdict['spec']['obs_flux'] = sunspec['flux']
		inputdict['spec']['obs_eflux'] = sunspec['err']
		inputdict['spec']['normspec'] = False
		inputdict['spec']['convertair'] = True
		# set an additional guassian prior on the instrument profile
		# inputdict['priordict']['Inst_R'] = {'gaussian':[47000.0,1000.0]}


if runphot:
	inputdict['phot'] = {}
	inputdict['photANNpath'] = '/Users/pcargile/Astro/GITREPOS/ThePayne/data/photANN/'

	if runmock:
		# MOCK PHOT
		phot = ([6.34406175,5.07894377,4.62912848,4.52416682,4.50593773,
			3.65719927,3.32468104,3.28475372,
			3.27660632,3.29905389,3.26541975])

		filterarr = (['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z',
			'2MASS_J','2MASS_H','2MASS_Ks',
			'WISE_W1','WISE_W2','WISE_W3',])

	else:
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

# set parameter for sampler
inputdict['sampler'] = {}
inputdict['sampler']['samplerbounds'] ='Nested'
inputdict['sampler']['samplemethod'] = 'slice'
inputdict['sampler']['npoints'] = 125
inputdict['sampler']['samplerbounds'] = 'multi'
inputdict['sampler']['flushnum'] = 100
inputdict['sampler']['delta_logz_final'] = 0.001
inputdict['sampler']['bootstrap'] = 0
# inputdict['sampler']['slices'] = 500

# set some flat priors for defining the prior volume
inputdict['priordict'] = {}
inputdict['priordict']['Teff']   = {'uniform':[3000.0,17000.0]}
inputdict['priordict']['log(g)'] = {'uniform':[1.0,5.5]}
inputdict['priordict']['[Fe/H]'] = {'uniform':[-4.0,0.5]}
inputdict['priordict']['[a/Fe]'] = {'uniform':[-0.2,0.6]}
inputdict['priordict']['Vrad'] = {'uniform':[-700.0,700.0]}
# gmi = inputdict['phot']['PS_g'][0]-inputdict['phot']['PS_i'][0]
# inputdict['priordict']['Vrot'] = (
# 	{'uniform':[0.0,400.0],
# 	'gaussian':[0.0,500.0/((5.0+gmi)**2.0)]}) 
inputdict['priordict']['Vrot']   = {'uniform':[0.0,5.0]}


# inputdict['priordict']['Dist']   = {'uniform':[1.0,200.0]}
# inputdict['priordict']['log(R)'] = {'uniform':[-0.1,0.1]}
inputdict['photscale'] = True
inputdict['priordict']['log(A)'] = {'uniform':[-3.0,7.0]}
inputdict['priordict']['Av']     = {'uniform':[0.0,1.0]}

# set an additional guassian prior on the instrument profile
inputdict['priordict']['Inst_R'] = (
	{'uniform':[27000.0,38000.0],
	'gaussian':[32000.0,1000.0]})

"""
# set parameter for sampler
inputdict['sampler'] = {}
inputdict['sampler']['samplerbounds'] ='Nested'
inputdict['sampler']['samplemethod'] = 'slice'
inputdict['sampler']['npoints'] = 250
inputdict['sampler']['samplerbounds'] = 'multi'
inputdict['sampler']['flushnum'] = 500
inputdict['sampler']['delta_logz_final'] = 0.001
inputdict['sampler']['bootstrap'] = 0

# set some flat priors for defining the prior volume
inputdict['priordict'] = {}
inputdict['priordict']['Teff']   = {'uniform':[5000.0,6500.0]}
inputdict['priordict']['log(g)'] = {'uniform':[4.0,5.0]}
inputdict['priordict']['[Fe/H]'] = {'uniform':[-0.1,0.1]}
inputdict['priordict']['[a/Fe]'] = {'uniform':[-0.1,0.1]}
inputdict['priordict']['Vrad']   = {'uniform':[-5.0,5.0]}
inputdict['priordict']['Vrot']   = {'uniform':[0.0,5.0]}
inputdict['priordict']['Inst_R'] = {'uniform':[25000.0,55000.0]}

# inputdict['priordict']['Dist']   = {'uniform':[5.0,20.0]}
inputdict['priordict']['Av']     = {'uniform':[0.0,1.0]}
# inputdict['priordict']['log(R)'] = {'uniform':[-0.1,0.1]}
"""

inputdict['output'] = 'demoout_multispec.dat'


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


