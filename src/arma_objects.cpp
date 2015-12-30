#include "arma_objects.h"
#include <iostream>

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
    } else if(args.length() == 2) {
        logger->debug() << "interpreting as (n_rows, n_cols) pair";
    } else if(args.length() > 2) {
        logger->error("ctor received too many arguments");
        throw Py::TypeError("too many arguments");
    }
    logger->debug("initialized");
}

arma_mat::~arma_mat()
{
    logger->debug("in arma_mat dtor");
}

/*
arma_mat::arma_mat() : m_mat()
{
    std::cout << "arma_mat default constructor" << std::endl;
}

arma_mat::arma_mat(arma::uword n_rows, arma::uword n_cols) : m_mat(n_rows, n_cols)
{
    std::cout << "arma_mat sized constructor";
    std::cout << "(" << n_rows << ", " << n_cols << ")" << endl;
}

arma_mat::arma_mat(arma::uword n_rows, arma::uword n_cols) : m_mat(n_rows, n_cols)
{
    std::cout << "arma_mat sized constructor";
    std::cout << "(" << n_rows << ", " << n_cols << ")" << endl;
}
*/

Py::Object arma_mat::n_rows() 
{
    std::cout << "called n_rows() " << std::endl;
    return Py::Long(long(m_mat.n_rows));
}

Py::Object arma_mat::n_cols() 
{
    std::cout << "called n_cols() " << std::endl;
    return Py::Long(long(m_mat.n_cols));
}

Py::Object arma_mat::n_elem() 
{
    std::cout << "called n_elem() " << std::endl;
    return Py::Long(long(m_mat.n_elem));
}
