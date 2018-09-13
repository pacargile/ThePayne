from Payne.predict import predictspec_multi,predictspec,predictsed
from astropy.table import Table,join,vstack ; import matplotlib.pyplot as plt ; import numpy as np

psedf = predictsed.FastPayneSEDPredict(
	usebands=['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z'],
	nnpath='/Users/pcargile/Astro/GITREPOS/ThePayne/data/photANN/')

indict = {}
indict['logt'] = np.log10(5000.0)
indict['logg'] = 4.5
indict['feh'] = 0.0
indict['afe'] = -0.2
indict['logA'] = 1.0
indict['av'] = 0.0


print(psedf.sed(**indict))