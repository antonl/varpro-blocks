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

    PYCXX_NOARGS_METHOD_DECL(arma_mat, n_rows)
    PYCXX_NOARGS_METHOD_DECL(arma_mat, n_cols)
    PYCXX_NOARGS_METHOD_DECL(arma_mat, n_elem)

private:
    arma::mat m_mat;
    std::shared_ptr<spdlog::logger> logger;
};

/*
class arma_vec : public Py::ExtensionType<arma_vec>
{
public:
    static void init_type(); 

    arma_vec();
    virtual ~arma_vec();

    Py::Long n_rows();
    Py::Long n_elem();
};
*/
