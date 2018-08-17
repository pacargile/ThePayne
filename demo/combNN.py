import h5py
import numpy as np
import glob

def smartsort(x):
	return float(x.split('_')[1].replace('w',''))


print('... Read in HDF5 files')
NNfilearr = glob.glob('test6_w*_*.h5')
NNfilearr.sort(key=smartsort)
th5list = [h5py.File(NNfile,'r') for NNfile in NNfilearr]
wavearr = [np.array(x['wavelength']) for x in th5list]
Rarr = [3000.0]

print('... Create output file')
outth5 = h5py.File('test6.h5','w')
outth5.create_dataset('resolution', data=Rarr, compression='gzip')
outth5.create_dataset('xmin',data=np.array([np.log10(2500.0),-1.0,-4.0,-0.2]))
outth5.create_dataset('xmax',data=np.array([np.log10(15000.0),5.5,0.5,0.6]))

for ii,(th5,ww) in enumerate(zip(th5list,wavearr)):
	outth5.create_dataset(
		'wavelength/{0}'.format(ii), 
		data=th5['wavelength'], 
		compression='gzip')
	for labelname in (
		['lin1.bias', 'lin1.weight', 'lin2.bias', 'lin2.weight', 'lin3.bias', 'lin3.weight']
		):
		inputlab = 'model_{0}_{1}/model/{2}'.format(ww[0],ww[-1],labelname)

		outth5.create_dataset('model/{0}/model/{1}'.format(ii,labelname),
			data=th5[inputlab],
			compression='gzip')

outth5.close()
