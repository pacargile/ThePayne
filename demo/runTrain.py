import Payne
TS = Payne.train.TrainSpec_multi(
	waverange=[5000.0,5000.05],
	numtrain=1000,
	niter=50000,
	epochs=25,
	resolution=80000.0,
	pixpernet=3,
	output='test3',
	logepoch=True,
	adaptivetrain=True,
	)
TS.run()
