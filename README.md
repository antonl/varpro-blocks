A test implementation of variable projection for global kinetic fitting in python/cython.

## Requirements

- cmake
- c compiler
- cython
- python 3.5 (not tested with anything else...)
- numpy

## Installing

Use ```cmake``` to build the extension. I'd recommend making a build directory
and compiling the code there (an out-of-source build).

```
# mkdir build; cd build
# cmake ..
... configuration happens...
# make 
# python setup.py install
```
