#pragma once

#include <armadillo>
#include <memory>
#include <tuple>
#include "spdlog/spdlog.h"

class response_block 
{
public:
    explicit response_block(const arma::vec& measured);
    virtual ~response_block();

    void update_model(const arma::vec p, bool update_jac=false);
    static constexpr auto name = "response_block";

    const std::tuple<arma::vec, arma::vec, arma::mat> get_yrJ() const;

    const std::tuple<arma::vec, arma::vec> get_params() const;
    const arma::vec get_target() const;
    const std::tuple<arma::mat, arma::umat, arma::mat, 
          arma::mat, arma::mat, arma::mat> get_internal() const;
    const std::tuple<arma::mat, arma::vec, arma::mat> get_svd() const;

protected:
    std::shared_ptr<spdlog::logger> log;

    virtual void evaluate_model(const arma::vec& p) = 0;
    virtual void evaluate_jacobian(const arma::vec& p) = 0;

    const arma::vec y; // measured response
    arma::uword M; // number of measurements
    arma::vec yh; // estimated response
    arma::vec resid; // residuals

    arma::mat Amat; // model matrix
    
    arma::umat jidx; // indexing matrix for sparse jacobian
    arma::mat mjac; // sparse matrix jacobian

    arma::uword feval, jeval; // evaluations of model function

    arma::mat J; // projected jacobian
    arma::mat dkc, dkrw; // cached terms used in varpro jacobian
    arma::vec alpha; // nonlinear parameter vector
    arma::vec beta; // linear parameter vector
    arma::mat U;
    arma::mat V;
    arma::vec s;

};

class exp_model : public response_block
{
public:
    explicit exp_model(const arma::vec& m, const arma::vec& t);
    virtual ~exp_model();
    static constexpr auto name = "exp_model";

protected:
    virtual void evaluate_model(const arma::vec&p);
    virtual void evaluate_jacobian(const arma::vec&p);
    const arma::vec tvec;
};
