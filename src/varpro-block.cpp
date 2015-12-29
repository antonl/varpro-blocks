#include "varpro-block.h"
#include "spdlog/spdlog.h"

#include <cstdlib>

#include <iostream>
#include <exception>

using namespace std;

auto logger = spdlog::stdout_logger_st("varpro-blocks");

template<> std::string get_size(const mat X) {
    std::stringstream s;
    s << "(" << X.n_rows << ", " << X.n_cols << ")";
    return s.str();
}

template<> std::string get_size(const vec X) {
    std::stringstream s;
    s << "(" << X.n_rows << ", )";
    return s.str();
}

response_block::response_block(const vec& m)
    : y(m)
{
    logger->debug("created response_block");
    M = m.n_rows;
    yh.copy_size(y);
    resid.copy_size(y);
}

const string response_block::get_name() const
{
    return "response_block";
}

void response_block::update_model(const vec p, bool update_jac)
{
    logger->debug("in response_block::update_model");
    _generate_model(p); // update the model matrix
    
    mat U, V;
    vec s;
    
    mat Ut, Vt; // for caching transposed stuff
    mat Apinv;

    bool success = svd_econ(U, s, V, Amat);
    if(!success)
        logger->warn("SVD decomposition failed");

    Ut = mat(U.t());
    Vt = mat(V.t());

    logger->debug("SVD decomposition succeeded");

    logger->debug("V size: {}, Ut size: {}, s size: {}", get_size(V), 
            get_size(Ut), get_size(s));

    logger->debug("Ut: \n{}", Ut);
    Apinv = V*diagmat(1/s)*Ut;
    beta = Apinv*y;
    yh = Amat*beta;
    resid = y - yh;
    logger->debug("Obtained betas: \n{}", beta);
    logger->debug("residuals: \n{}", resid);

    if (update_jac)
        _generate_jacobian(p);
}

void response_block::_generate_model(const vec& p)
{
}

void response_block::_generate_jacobian(const vec& p)
{
}

single_exp_model::single_exp_model(const vec &m, const vec &t)
    : response_block(m), tvec(t)
{
    logger->debug("in single_exp_model constructor");
    if (tvec.size() != m.size())
        throw std::logic_error("t and measured vectors must match size");

    P = 1;
    N = 2;

    // Model of the form A + B exp(-k t)
    Amat = mat(M, N, arma::fill::zeros);
    jidx = {{0, 1}};
    mjac = mat(M, 1);
}

const string single_exp_model::get_name() const 
{
    return "single_exp_model";
}

void single_exp_model::_generate_model(const vec& p)
{
    double t;
    for(uword m=0; m < M; m++) {
        t = tvec(m);
        Amat(m, 0) = 1.;
        Amat(m, 1) = std::exp(-t*p(0));
    }
}

void single_exp_model::_generate_jacobian(const vec& p)
{
    double t;
    for(uword m=0; m < M; m++) {
        t = tvec(m);
        mjac(m, 0) = -t*std::exp(-t*p(0));
    }
}

const vec& single_exp_model::get_t() const
{
    return tvec;
}
