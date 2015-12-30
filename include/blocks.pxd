from libcpp cimport bool
from libcpp.string cimport string
cimport arma

cdef extern from "varpro-block.h" nogil:
    cdef cppclass response_block:
        response_block(const arma.vec& measured)
        void update_model(const arma.vec& p, bool update_jac = False)
        const string get_name() const

        arma.vec y
        arma.vec yh
        arma.vec resid
