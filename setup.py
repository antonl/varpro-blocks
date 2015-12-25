from setuptools import setup
from distutils.sysconfig import get_python_lib
import glob
import os
import sys

if os.path.exists('readme.rst'):
    print("""The setup.py script should be executed from the build directory.

Please see the file 'readme.rst' for further instructions.""")
    sys.exit(1)

setup(
    name = "varpro",
    package_dir = {'': 'src'},
    data_files = [(get_python_lib(), glob.glob('src/*.so'))],
    author = 'Anton Loukianov',
    description = 'An experiment for performing regression using variable projection on multiple responses at the same time, "Global Fitting."',
    license = 'Apache',
    keywords = 'regression numpy cython cmake statistics',
    url = 'https://github.com/antonl/varpro',
    test_require = ['nose'],
    zip_safe = False,
    )
