#pragma once

#include <memory>
#include <armadillo>
#include "CXX/Objects.hxx"
#include "CXX/Extensions.hxx"
#include "spdlog/spdlog.h"

class arma_mat : public Py::PythonClass<arma_mat>
{
public:
    static void init_type(void);

    arma_mat(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds);
    virtual ~arma_mat();

    Py::Object n_rows();
    Py::Object n_cols();
    Py::Object n_elem();
    Py::Object to_numpy();

    PYCXX_NOARGS_METHOD_DECL(arma_mat, n_rows)
    PYCXX_NOARGS_METHOD_DECL(arma_mat, n_cols)
    PYCXX_NOARGS_METHOD_DECL(arma_mat, n_elem)
    PYCXX_NOARGS_METHOD_DECL(arma_mat, to_numpy)

private:
    arma::mat m_mat;
    std::shared_ptr<spdlog::logger> logger;

    void _construct_from_ndarray(const Py::Object nparray);
};

class arma_vec : public Py::PythonClass<arma_vec>
{
public:
    static void init_type(void);

    arma_vec(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds);
    virtual ~arma_vec();

    Py::Object n_rows();
    Py::Object n_elem();
    Py::Object to_numpy();

    PYCXX_NOARGS_METHOD_DECL(arma_vec, n_rows)
    PYCXX_NOARGS_METHOD_DECL(arma_vec, n_elem)
    PYCXX_NOARGS_METHOD_DECL(arma_vec, to_numpy)

private:
    arma::vec m_vec;
    std::shared_ptr<spdlog::logger> logger;

    void _construct_from_ndarray(const Py::Object nparray);
};
