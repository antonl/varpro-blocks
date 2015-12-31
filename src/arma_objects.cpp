#include "arma_objects.h"
#include <iostream>
#include <vector>

#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

void arma_mat::init_type(void)
{
    spdlog::stdout_logger_mt("arma.mat");
    auto logger = spdlog::get("arma");
    logger->debug("initialized arma_mat type");
    
    behaviors().name("mat");
    behaviors().doc("armadillo mat class");

    PYCXX_ADD_NOARGS_METHOD(n_cols, n_cols, "return number of columns in matrix");
    PYCXX_ADD_NOARGS_METHOD(n_rows, n_rows, "return number of rows in matrix");
    PYCXX_ADD_NOARGS_METHOD(n_elem, n_elem, "return number of elements in matrix");
    PYCXX_ADD_NOARGS_METHOD(to_numpy, to_numpy, "return matrix as numpy array");
    behaviors().readyType();
}

void arma_vec::init_type(void)
{
    spdlog::stdout_logger_mt("arma.vec");
    auto logger = spdlog::get("arma");
    logger->debug("initialized arma_vec type");
    
    behaviors().name("vec");
    behaviors().doc("armadillo vec class");

    PYCXX_ADD_NOARGS_METHOD(n_rows, n_rows, "return number of rows in vector");
    PYCXX_ADD_NOARGS_METHOD(n_elem, n_elem, "return number of elements in vector");
    PYCXX_ADD_NOARGS_METHOD(to_numpy, to_numpy, "return vector as numpy array");
    behaviors().readyType();
}

arma_mat::arma_mat(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds):
    Py::PythonClass<arma_mat>::PythonClass(self, args, kwds),
    logger(spdlog::get("arma.mat"))
{
    logger->set_level(spdlog::level::debug);
    logger->debug("in arma_mat ctor");

    if(args.length() == 1) {
        logger->debug() << "interpreting as numpy ndarray";
        _construct_from_ndarray(args[0]);
    } else if(args.length() == 2) {
        logger->debug() << "interpreting as (n_rows, n_cols) pair";
        m_mat = arma::mat(Py::Long(args[0]), Py::Long(args[1]));
    } else {
        logger->error("ctor received incorrect number of arguments ({})", args.length());
        throw Py::TypeError("wrong number of arguments");
    }
    logger->debug("arma_mat initialized");
}

arma_vec::arma_vec(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds):
    Py::PythonClass<arma_vec>::PythonClass(self, args, kwds),
    logger(spdlog::get("arma.vec"))
{
    logger->set_level(spdlog::level::debug);
    logger->debug("in arma_vec ctor");

    if(args.length() == 1) {
        if(PyArray_CheckExact(args[0].ptr())) {
            logger->debug("interpreting as np array");
            _construct_from_ndarray(args[0]);
        } else {
            logger->debug("interpreting as dimension");
            m_vec = arma::vec(Py::Long(args[0]), arma::fill::zeros);
        }
    } else {
        logger->error("ctor received incorrect number of arguments ({})", args.length());
        throw Py::TypeError("wrong number of arguments");
    }
    logger->debug("arma_vec initialized");
}

arma_mat::~arma_mat()
{
    logger->debug("in arma_mat dtor");
}

arma_vec::~arma_vec()
{
    logger->debug("in arma_vec dtor");
}

arma_mat::operator arma::mat()
{
    return m_mat;
}

arma_vec::operator arma::vec()
{
    return m_vec;
}

void arma_mat::_construct_from_ndarray(const Py::Object array) 
{
    if(!PyArray_CheckExact(array.ptr())) {
        logger->error("expected numpy array type as input");
        throw Py::TypeError("expected numpy array type as input");
    }

    PyArrayObject *ap1 = reinterpret_cast<PyArrayObject *>(array.ptr());
    if(!(PyArray_NDIM(ap1) == 2)) {
        logger->error("expected 2-d input");
    }

    PyObject *contig = PyArray_FROM_OTF(array.ptr(), NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    Py::Object c = Py::asObject(contig);
    PyArrayObject *ap2 = reinterpret_cast<PyArrayObject *>(c.ptr());
    const double *cdata = reinterpret_cast<double *>(PyArray_DATA(ap2));
    arma::uword i = PyArray_DIM(ap2, 0);
    arma::uword j = PyArray_DIM(ap2, 1);
    m_mat = arma::mat(cdata, i, j);
}

void arma_vec::_construct_from_ndarray(const Py::Object array) 
{
    if(!PyArray_CheckExact(array.ptr())) {
        logger->error("expected numpy array type as input");
        throw Py::TypeError("expected numpy array type as input");
    }

    PyArrayObject *ap1 = reinterpret_cast<PyArrayObject *>(array.ptr());
    if(!(PyArray_NDIM(ap1) == 1)) {
        logger->error("expected 1-d input");
    }

    PyObject *contig = PyArray_FROM_OTF(array.ptr(), NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    Py::Object c = Py::asObject(contig);
    PyArrayObject *ap2 = reinterpret_cast<PyArrayObject *>(c.ptr());
    const double *cdata = reinterpret_cast<double *>(PyArray_DATA(ap2));
    arma::uword i = PyArray_DIM(ap2, 0);
    m_vec = arma::vec(cdata, i);
}

Py::Object arma_mat::n_rows() 
{
    logger->debug("in arma_mat::n_rows(); returning {}", m_mat.n_rows);
    return Py::Long(m_mat.n_rows);
}

Py::Object arma_vec::n_rows() 
{
    logger->debug("in arma_vec::n_rows(); returning {}", m_vec.n_rows);
    return Py::Long(m_vec.n_rows);
}

Py::Object arma_mat::n_cols() 
{
    logger->debug("in arma_mat::n_cols(); returning {}", m_mat.n_cols);
    return Py::Long(m_mat.n_cols);
}

Py::Object arma_mat::n_elem() 
{
    logger->debug("in arma_mat::n_elem(); returning {}", m_mat.n_elem);
    return Py::Long(m_mat.n_elem);
}

Py::Object arma_vec::n_elem() 
{
    logger->debug("in arma_vec::n_elem(); returning {}", m_vec.n_elem);
    return Py::Long(m_vec.n_elem);
}

Py::Object arma_mat::to_numpy()
{
    logger->debug("in arma_mat::to_numpy");
    // Warning! the largest size is limited by int!
    std::vector<long int> dims = {long(m_mat.n_rows), long(m_mat.n_cols)};
    auto desc = PyArray_DescrFromType(NPY_DOUBLE);
    PyObject *tmp = PyArray_NewFromDescr(&PyArray_Type, desc, 2, &(dims[0]), NULL, m_mat.memptr(), NPY_ARRAY_FARRAY_RO, NULL);
    return Py::asObject(tmp);
}

Py::Object arma_vec::to_numpy()
{
    logger->debug("in arma_vec::to_numpy");
    // Warning! the largest size is limited by int!
    std::vector<long int> dims = {long(m_vec.n_rows)};
    auto desc = PyArray_DescrFromType(NPY_DOUBLE);
    PyObject *tmp = PyArray_NewFromDescr(&PyArray_Type, desc, 1, &(dims[0]), NULL, m_vec.memptr(), NPY_ARRAY_FARRAY_RO, NULL);
    return Py::asObject(tmp);
}
