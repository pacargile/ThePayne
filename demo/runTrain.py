import Payne
TS = Payne.train.TrainSpec_V2(
	waverange=[5000.0,5000.1],
	numtrain=30,
	niter=1000,
	epochs=10,
	resolution=10000.0,
	output='test_train.h5',
	)
TS.run()
