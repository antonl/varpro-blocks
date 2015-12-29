from libcpp cimport bool
cimport numpy as np

cdef extern from "<armadillo>" namespace "arma" nogil:
    cdef cppclass mat:
        mat(double *aux_mem, int n_rows, int n_cols, bool copy_aux_mem, \
                bool strict) nogil except +
        mat(double *aux_mem, int n_rows, int n_cols) nogil except +
        mat(int n_rows, int n_cols) nogil except +
        mat() nogil except +
        
        int n_rows
        int n_cols
        int n_elem
        double *memptr()
        void print()

    cdef cppclass vec:
        vec(double * aux_mem, int number_of_elements, bool copy_aux_mem, \
                bool strict) nogil except +
        vec(double * aux_mem, int number_of_elements) nogil except +
        vec(int) nogil except +
        vec() nogil except +
        int n_rows
        int n_cols
        int n_elem
        double *memptr()
        void print()

cdef mat* np2mat(np.ndarray[ndim=2, dtype=np.float64_t] X)
cdef vec* np2vec(np.ndarray[ndim=1, dtype=np.float64_t] X)

cdef np.ndarray mat2np(mat *X)
cdef np.ndarray vec2np(vec *X)

cdef extern from "varpro-block.h" nogil:
    cdef cppclass response_block:
        response_block(const mat& m)
        update_model(vec p, bool update_jac)
