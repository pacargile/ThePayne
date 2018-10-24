from Payne.predict import predictspec_multi
pp = predictspec_multi.PayneSpecPredict('test1.h5')
print(pp.NN['wavelength'])
print(pp.predictspec([5770.0,4.44,0.0,0.0]))
