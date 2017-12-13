#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

    
setup(
    name="Payne",
    url="https://github.com/pacargile/ThePayne.git",
    version="1.0",
    author="Phillip Cargile",
    author_email="pcargile@cfa.harvard.edu",
    packages=["Payne",
              "Payne.fitting",
              "Payne.predict",
              "Payne.testing",
              "Payne.train",
              "Payne.utils"],
    license="LICENSE",
    description="The Payne: ANN based stellar spectra and SED prediction and modeling code.",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "dynesty", "torch"],
)

# write top level __init__.py file with the correct absolute path to package repo
toplevelstr = ("""try:
    from ._version import __version__
except(ImportError):
    pass

from . import fitting
from . import predict
from . import train
from . import testing
from . import utils"""
)

with open('Payne/__init__.py','w') as ff:
  ff.write(toplevelstr)
  ff.write('\n')
  ff.write("""__abspath__ = '{0}/'\n""".format(os.getcwd()))