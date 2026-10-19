#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='medley',
    version='0.0.1',
    license="BSD-3-Clause",
    entry_points={
         'console_scripts': [
             'medley = medley.cli:main',
             'npz_to_cbf = medley.npz_to_cbf:main',
          ]
    },
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'ase',
        'abtem==1.0.5',
        'fabio',
    ],
)
