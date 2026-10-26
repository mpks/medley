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
             'medley.invert_cif = medley.invert_cif:main',
             'medley.compare_pairs = medley.compare_pairs:main',
             'medley.npz_to_png = medley.npz_to_png:main',
             'medley.integrate = medley.integrate_npz:main',
             'medley.compare_hkl = medley.compare_hkl:main',
          ]
    },
    packages=find_packages(),
    install_requires=[
        'numpy<2',
        'scipy<1.17',
        'matplotlib',
        'ase',
        'abtem==1.0.5',
        'fabio',
    ],
)
