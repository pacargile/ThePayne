import Payne
TS = Payne.train.TrainSpec_V2(
	waverange=[5000.0,5001.0],
	numtrain=150,
	niter=3000,
	epochs=30,
	resolution=10000.0,
	output='test_train.h5',
	)
TS.run()
