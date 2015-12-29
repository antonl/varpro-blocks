import numpy as np
cimport numpy as np

cdef mat* np2mat(np.ndarray[ndim=2, dtype=np.float64_t] X):
    if not (X.flags.f_contiguous or X.flags.owndata):
        X = X.copy(order="F") 
    
    cdef mat *converted = new mat(<double *>X.data, X.shape[0], X.shape[1])
    return converted # must remember to delete this object!

cdef vec* np2vec(np.ndarray[ndim=1, dtype=np.float64_t] X):
    if not (X.flags.f_contiguous or X.flags.owndata):
        X = X.copy(order="F") 
    
    cdef vec *converted = new vec(<double *>X.data, X.shape[0])
    return converted # must remember to delete this object!

cdef np.ndarray mat2np(mat *X):
    cdef double [:] mv = <double [:X.n_elem]> X.memptr()
    cdef np.ndarray converted = np.array(mv, copy=True, order='F')
    return converted.reshape(X.n_rows, X.n_cols)

cdef np.ndarray vec2np(vec *X):
    cdef double [:] mv = <double [:X.n_elem]> X.memptr()
    cdef np.ndarray converted = np.array(mv, copy=True, order='F')
    return converted

def make_response(np.ndarray[ndim=1, dtype=np.float64_t] y):
    cdef vec *yvec = np2vec(y)

def test_roundtrip_vec(np.ndarray[ndim=1, dtype=np.float64_t] X):
    cdef vec *xvec
    cdef np.ndarray result

    try:
        xvec = np2vec(X)
        xvec.print()
        result = vec2np(xvec)
    finally:
        del xvec
    return result

def test_roundtrip_mat(np.ndarray[ndim=2, dtype=np.float64_t] X):
    cdef mat *xmat
    cdef np.ndarray result

    try:
        xmat = np2mat(X)
        xmat.print()
        result = mat2np(xmat)
    finally:
        del xmat
    return result

