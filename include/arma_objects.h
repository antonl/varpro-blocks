#pragma once

#include <armadillo>
#include "CXX/Objects.hxx"
#include "CXX/Extensions.hxx"

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
