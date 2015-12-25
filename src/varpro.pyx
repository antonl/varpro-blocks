from __future__ import print_function
from cpython cimport bool

DEF MODEL_DEBUG = True

import numpy as np
cimport numpy as np

from scipy.linalg import svd, diagsvd
from libc.math cimport exp

def test(s):
    print(s)

cdef class ResponseBlock:
    '''represents a sub-block of a global model function for an individual response
    
    '''
    
    cdef double [:] y, yh, resid, beta
    cdef double [:, :] U, Ut, V, Vt, Sinv
    cdef double [:, :] model_matrix, Apinv
    
    cdef double [:, :] mjac, jac
    cdef double [:, :] dkrw, dkc

    cdef unsigned int [:, :] jidx

    cdef unsigned int M,N,K,P
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

    property mjacobian:
        def __get__(self):
            return np.asarray(self.mjac)

    property jacobian:
        def __get__(self):
            return np.asarray(self.jac)

    def __init__(self, measured_response, n_nonlinear_params, n_linear_params):
        if len(measured_response.shape) > 1:
            raise ValueError('expected 1-d measured response vector')
        
        # define size of block
        self.y = measured_response
        self.yh = np.empty_like(measured_response) # estimated response
        self.resid = np.empty_like(measured_response) # residuals
        self.M = measured_response.shape[0]
        self.P = n_nonlinear_params
        self.N = n_linear_params
        self.beta = np.empty((self.N), dtype=np.float64)
        
        self.model_matrix = np.empty((self.M, self.N), dtype=np.float64)
        
        # preallocate space for running SVD
        self.K = min(self.M, self.N)
        self.U = np.empty((self.M, self.K), dtype=np.float64)
        self.Vt = np.empty((self.K, self.N), dtype=np.float64)

        # for caching inverses
        self.Ut = np.empty((self.K, self.M), dtype=np.float64)
        self.V = np.empty((self.N, self.K), dtype=np.float64)

        self.Sinv = np.empty((self.K, self.K), dtype=np.float64)
        self.Apinv = np.empty((self.N, self.M), dtype=np.float64)

        # preallocate space for jacobian calculations
        # self._init_jac_storage(2) # remember to call this in subclass

        self.model_name = u'ResponseBlock'

    cdef void _init_jac_storage(self, unsigned int Q):
        # initialize all of the required storage given the number of
        # nonzero columns in the Jacobian
        self.jidx = np.zeros((Q, 2), dtype=np.uint32)

        self.mjac = np.zeros((self.M, Q), dtype=np.float64)
        self.dkc = np.zeros((self.M, Q), dtype=np.float64)
        self.dkrw = np.zeros((self.N, Q), dtype=np.float64)

    def update_model(self, double [:] p, bool eval_jac=False):
        self._generate_model_matrix(p)
        self._evaluate_model(p)

        if eval_jac:
            self._generate_model_jacobian(p)
            self._evaluate_jacobian(p)
    
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
            
        self.U, s, self.Vt = svd(self.model_matrix, compute_uv=True, 
                full_matrices=False)

        IF MODEL_DEBUG:
            print('svd...', end='')
        self.Sinv = np.diag(1/s)
        IF MODEL_DEBUG:
            print('sigma...', end='')
        self.V = np.transpose(self.Vt)
        self.Ut = np.transpose(self.U)
        np.dot(self.V, np.dot(self.Sinv, self.Ut), out=np.asarray(self.Apinv))
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

    cdef void _generate_model_jacobian(self, double [:] p):
        raise NotImplementedError('do not have a model in base class')
    
    cdef void _evaluate_jacobian(self, double [:] p):
        cdef unsigned int i, j, basis_no, param_no

        IF MODEL_DEBUG:
            print('Evaluating varpro jacobian...', end='')
        
        for i in range(self.jidx.shape[0]):
            basis_no = self.jidx[i, 0]
            param_no = self.jidx[i, 1]

            IF MODEL_DEBUG:
                print('scaling ({:d},{:d})...'.format(basis_no, param_no), end='')

            for j in range(self.M):
                # i here is a proxy for param_no since that is sparse
                self.dkc[j, i] = self.mjac[j, i]*self.beta[basis_no]
                self.dkrw[basis_no, i] = self.mjac[j, i]*self.resid[j]

        # now we have a dense representation of the rescaled jacobian
        IF MODEL_DEBUG:
            print('A...', end='')
        A = np.subtract(self.dkc, np.dot(self.U, np.dot(self.Ut, self.dkc)))
        IF MODEL_DEBUG:
            print('B...', end='')
        B = np.dot(self.U, np.dot(self.Sinv, np.dot(self.Vt, self.dkrw)))

        IF MODEL_DEBUG:
            print('J...', end='')
        J = np.add(A, B) # times -1 for true jacobian

        # merge terms that correspond to the same nonlinear parameter
        IF MODEL_DEBUG:
            print('merging...', end='')
        self.jac = np.zeros((self.M, self.P), dtype=np.float64)
        for i in range(self.jidx.shape[0]):
            basis_no = self.jidx[i, 0]
            param_no = self.jidx[i, 1]

            for j in range(self.M):
                self.jac[j, param_no] += -1.*J[j, i]
        IF MODEL_DEBUG:
            print('done')

    def __repr__(self):
        return \
    '''{:s}(measured_response={!r}, n_linear_params={:d})'''.format(self.model_name, self.y, self.N)
    
    def __str__(self):
        return '''{:s}, {:d}x{:d}'''.format(self.model_name, self.M, self.N)
    
cdef class SingleExpBlock(ResponseBlock):
    cdef double [:] t
    
    def __init__(self, measured_response, t):
        super(SingleExpBlock, self).__init__(measured_response,
                n_nonlinear_params=1, n_linear_params=2)
        
        if len(t.shape) > 1:
            raise ValueError('expected 1-d t vector')
        
        # initialize storage for jacobian matrix
        self._init_jac_storage(1)

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

    cdef void _generate_model_jacobian(self, double [:] p):
        cdef unsigned int i
        cdef double t

        # assemble index array
        self.jidx[0, 0] = 1
        self.jidx[0, 1] = 0

        for i in range(self.M):
            t = self.t[i]
            self.mjac[i, 0] = -t*exp(-t*p[0])
