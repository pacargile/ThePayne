# The Payne
=====

Version 0.1 (beta)

Artificial Neural-Net compression and fitting of ab initio synthetic spectral grids. The code has the following functionality:

* Using PyTorch to train an artificial neural-network on individual pixels in a grid of spectra and/or photometry

* Testing and Validation of a trained ANN to evaluate its precision and calculate the covariance in the predictions

* Fit observed spectra and/or photometry using trained ANN coupled with Bayesian sampling.

The general algorithm framework for The Payne is given in [Ting, Y.-S., Conroy, C., Rix, H.-W., & Cargile, P. 2018, ApJ, submitted](https://ui.adsabs.harvard.edu/#abs/2018arXiv180401530T). Further details on building ab initio models of synthetic stellar spectra can be found in [Ting, Y.-S., Conroy, C., & Rix, H.-W. 2016, ApJ, 826, 83](http://adsabs.harvard.edu/abs/2016ApJ...826...83T) and [Ting, Y.-S., Rix, H.-W., Conroy, C., Ho, A.~Y.~Q., & Lin, J. 2017, ApJL, 849, L9](http://adsabs.harvard.edu/abs/2017ApJ...849L...9T). This code follows the general approach outlined in these papers, with additional changes in the fitting and training procedures. Please cite these references if this code is used for any academic purposes.

The current version of The Payne code is under development and not yet ready for wide distribution. Anyone interested in using this version of the code, please first contact <pcargile@cfa.harvard.edu>.

***The Payne is named in honor of Cecilia Payne-Gaposchkin, one of the great scientist of the 20th century and a pioneering in stellar astrophysicist. In the 1920s she derived the cosmic abundance of the elements from stellar spectra, including determining the composition of the Sun, and demonstrated for the first time the chemical homogeneity of the universe. Cecilia Payne-Gaposchkin achieved two Harvard firsts: she became the first female professor, and the first woman to become department chair.***

<https://en.wikipedia.org/wiki/Cecilia_Payne-Gaposchkin>


Authors
-------

* **Phillip Cargile** (Harvard)
* **Yuan-Sen Ting** (Carnegie-Princeton-IAS Fellow)

See [Authors](authors.rst) for a full list of contributors to The Payne.

Installation
------
```
cd <install_dir>
git clone https://github.com/pacargile/ThePayne.git
cd ThePayne
python setup.py install (--user)
```

Then in Python
```python
import Payne
```

The Payne is pure python.
See the [tutorial](demo/) for fitting a solar mock with photometric and spectroscopic data.

The user either need to train a new ANN for an observed spectrum and photometry. Or contact <pcargile@cfa.harvard.edu> to get the latest C3K based ANN. 


License
--------

Copyright 2018. The Payne is open-source software released under 
the MIT License. See the file ``LICENSE`` for details.

