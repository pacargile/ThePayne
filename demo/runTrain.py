import Payne
TS = Payne.train.TrainSpec(
	waverange=[5000.0,5001.0],
	numtrain=50,
	niter=3000,
	epochs=10,
	resolution=1000.0,
	output='test_train.h5',
	)
TS.run()
