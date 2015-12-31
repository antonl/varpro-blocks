#pragma once

#include <type_traits>
#include <memory>
#include <armadillo>
#include <string>
#include "CXX/Objects.hxx"
#include "CXX/Extensions.hxx"
#include "spdlog/spdlog.h"

class cvarpro_block
{
public:
    explicit cvarpro_block(const arma::vec measured);
    virtual ~cvarpro_block();

    cvarpro_block(const cvarpro_block& other) = delete;
    cvarpro_block& operator=(const cvarpro_block& other) = delete;

    void update_model(const arma::vec p, bool update_jac=false);

    constexpr static auto name = "cvarpro_block";
protected:
    const arma::vec y; // measured response
    arma::vec yh; // estimated response
    arma::vec resid; // residuals

    arma::uword M; // number of measurements in this block
    arma::uword P; // number of nonlinear parameters
    arma::uword N; // number of linear parameters

    arma::vec alpha; // cached nonlinear parameter vector
    arma::vec beta; // cached linear parameter vector

    arma::mat Amat; // model matrix

    arma::umat jidx; // jacobian indexing matrix
    arma::mat mjac; // nonzero jacobian matrix columns

    virtual void _generate_model_matrix(const arma::vec p) = 0;
    virtual void _generate_jacobian_matrix(const arma::vec p) = 0;

    std::shared_ptr<spdlog::logger> logger;
private:
    cvarpro_block() = default;
};

class single_exp_block : public cvarpro_block
{
public:
    explicit single_exp_block(const arma::vec measured, const arma::vec t);
    virtual ~single_exp_block();
protected:
    virtual void _generate_model_matrix(const arma::vec p);
    virtual void _generate_jacobian_matrix(const arma::vec p);
private:
    const arma::vec m_t;
};

// template wrapper for cvarpro_block types
template <typename T> class varpro_block: 
    public Py::PythonClass<varpro_block<T>>
{
public:
    static void init_type();

    varpro_block(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds);
    virtual ~varpro_block();
private:
    std::shared_ptr<spdlog::logger> logger;
    T m_block;

    static_assert(std::is_base_of<cvarpro_block, T>::value, "varpro_block must wrap a cvarpro_block type");
};

template class varpro_block<single_exp_block>;

