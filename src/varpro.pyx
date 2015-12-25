from __future__ import print_function

DEF MODEL_DEBUG = True

import numpy as np
cimport numpy as np

from scipy.linalg import svd, diagsvd
from libc.math cimport exp

def test(s):
    print(s)

def comparison_fmt(x):
    return '{:0.5g}'.format(x)

cdef class ResponseBlock:
    '''represents a sub-block of a global model function for an individual response
    
    '''
    
    cdef double [:] y, yh, resid, beta
    cdef double [:, :] U, Vh, Sinv
    cdef double [:, :] model_matrix, Apinv

    cdef unsigned int M,N,K
    cdef unicode model_name
    
    property residuals:
        def __get__(self):
            return np.asarray(self.resid)
        
    property beta:
        def __get__(self):
            return np.asarray(self.beta)
    
    property measured:
        def __get__(self):
            return np.asarray(self.y)
    
    property estimated:
        def __get__(self):
            return np.asarray(self.yh)

    def __init__(self, measured_response, n_linear_params):
        if len(measured_response.shape) > 1:
            raise ValueError('expected 1-d measured response vector')
        
        # define size of block
        self.y = measured_response
        self.yh = np.empty_like(measured_response) # estimated response
        self.resid = np.empty_like(measured_response) # residuals
        self.M = measured_response.shape[0]
        self.N = n_linear_params
        self.beta = np.empty((self.N), dtype=np.float64)
        
        self.model_matrix = np.empty((self.M, self.N), dtype=np.float64)
        
        # preallocate space for running SVD
        self.K = min(self.M, self.N)
        self.U = np.empty((self.M, self.K), dtype=np.float64)
        self.Vh = np.empty((self.K, self.N), dtype=np.float64)
        self.Sinv = np.empty((self.K, self.K), dtype=np.float64)
        self.Apinv = np.empty((self.N, self.M), dtype=np.float64)
        
        # preallocate space for calculating the Jacobian
        self.model_name = u'ResponseBlock'

    def update_model(self, double [:] p):
        self._generate_model_matrix(p)
        self._evaluate_model(p)
    
    cdef void _generate_model_matrix(self, double [:] p):
        cdef unsigned int i,j
        
        IF MODEL_DEBUG:
            print('Calculating default model matrix...', end='')

        for i in range(self.M):
            for j in range(self.N):
                self.model_matrix[i, j] = -1.

        IF MODEL_DEBUG:
            print('done')

    cdef void _evaluate_model(self, double [:] p):
        cdef unsigned int i,j,k
        cdef double t
        
        IF MODEL_DEBUG:
            print('Estimating linear parameters...', end='')
            
        self.U, s, self.Vh = svd(self.model_matrix, compute_uv=True, 
                full_matrices=False)

        IF MODEL_DEBUG:
            print('svd...', end='')
        self.Sinv = np.diag(1/s)
        IF MODEL_DEBUG:
            print('sigma...', end='')
        np.dot(np.transpose(self.Vh), np.dot(self.Sinv, np.transpose(self.U)),
                out=np.asarray(self.Apinv))
        IF MODEL_DEBUG:
            print('Apinv...', end='')
        np.dot(self.Apinv, self.y, out=np.asarray(self.beta))
        IF MODEL_DEBUG:
            print('beta...', end='')
        
        np.dot(self.model_matrix, self.beta, out=np.asarray(self.yh))
        IF MODEL_DEBUG:
            print('yh...', end='')
        np.subtract(self.yh, self.y, out=np.asarray(self.resid))
        
        IF MODEL_DEBUG:
            print('done')
            print('Smallest singular value {:.5g}'.format(s[-1]))
    
    cdef void calculate_jacobian(self, double [:] p):
        # virtual implementation, do nothing
        pass
    
    IF MODEL_DEBUG:
        def test(self):
            fmt = {'float_kind': comparison_fmt}
            print('Some data: {:s}'.format(np.array2string(np.asarray(self.y), formatter=fmt)))
    
    def __repr__(self):
        return \
    '''{:s}(measured_response={!r}, n_linear_params={:d})'''.format(self.model_name, self.y, self.N)
    
    def __str__(self):
        return '''{:s}, {:d}x{:d}'''.format(self.model_name, self.M, self.N)
    
cdef class SingleExpBlock(ResponseBlock):
    cdef double [:] t
    
    def __init__(self, measured_response, t):
        super(SingleExpBlock, self).__init__(measured_response, 2)
        
        if len(t.shape) > 1:
            raise ValueError('expected 1-d t vector')

        self.t = t
        self.model_name = u'SingleExpBlock'

    cdef void _generate_model_matrix(self, double [:] p):
        cdef unsigned int i
        cdef double t
        
        IF MODEL_DEBUG:
            print('Calculating SingleExpBlock model matrix...', end='')

        for i in range(self.M):
            t = self.t[i]
            self.model_matrix[i, 0] = 1.
            self.model_matrix[i, 1] = exp(-t*p[0])

        IF MODEL_DEBUG:
            print('done')
