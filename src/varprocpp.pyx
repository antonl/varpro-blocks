from cython.operator cimport dereference as deref
from libcpp cimport bool
import numpy as np
cimport numpy as np
from arma cimport vec, mat, np2vec, np2mat, mat2np, vec2np
from blocks cimport response_block

cdef class ResponseBlock:
    cdef response_block *thisptr

    property y:
        def __get__(self):
            return vec2np(&self.thisptr.y)

    property yh:
        def __get__(self):
            return vec2np(&self.thisptr.yh)

    def __cinit__(self, np.ndarray y):
        cdef vec *yvec = np2vec(y)

        try:
            self.thisptr = new response_block(yvec)
        finally:
            del yvec

    cpdef void update_model(self, np.ndarray[ndim=1, dtype=np.float64_t] p,
            bool update_jac = False):
        pass
