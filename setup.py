#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of the GW DetChar python (gwdetchar) package.
#
# gwdetchar is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gwdetchar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gwdetchar.  If not, see <http://www.gnu.org/licenses/>.

"""Setup the gwdetchar package
"""

from __future__ import print_function

import glob
import os.path
import sys

from setuptools import (setup, find_packages)

# local setup utilities
from _setup_utils import CMDCLASS as cmdclass

# set basic metadata
PACKAGENAME = 'gwdetchar'
DISTNAME = 'gwdetchar'
AUTHOR = 'Duncan Macleod'
AUTHOR_EMAIL = 'duncan.macleod@ligo.org'
LICENSE = 'GPLv3'

# -- versioning ---------------------------------------------------------------

import versioneer
__version__ = versioneer.get_version()

# -- dependencies -------------------------------------------------------------

# build
setup_requires = [
    'libsass',
    'jsmin',
]

# run
install_requires = [
    'six',
    'numpy>=1.10',
    'scipy>=0.18.1',
    'matplotlib>=2.0.0',
    'astropy>=1.2',
    'gwpy>=0.12.0',
    'lscsoft-glue>=1.60.0',
    'gwtrigfind',
    'sklearn',
]

# test
if 'test' in sys.argv:
    setup_requires.append('pytest-runner')
tests_require = [
    'pytest',
]

# -- run setup ----------------------------------------------------------------

packagenames = find_packages()
scripts = glob.glob(os.path.join('bin', '*'))

setup(name=DISTNAME,
      provides=[PACKAGENAME],
      version=__version__,
      description=None,
      long_description=None,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url='https://github.com/ligovirgo/gwdetchar',
      packages=packagenames,
      include_package_data=True,
      cmdclass=cmdclass,
      scripts=scripts,
      setup_requires=setup_requires,
      install_requires=install_requires,
      use_2to3=True,
      classifiers=[
          'Programming Language :: Python',
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      ],
      )
