#pragma once

#include <armadillo>
#include <memory>
#include <tuple>
#include "spdlog/spdlog.h"

typedef std::tuple<const arma::vec, const arma::mat> yJ_pair;

class response_block 
{
public:
    explicit response_block(const arma::vec& measured);
    virtual ~response_block();

    void update_model(const arma::vec p, bool update_jac=false);
    static constexpr auto name = "response_block";

    const yJ_pair get_yJ() const;

protected:
    virtual void evaluate_model(const arma::vec& p);
    virtual void evaluate_jacobian(const arma::vec& p);

    const arma::vec y; // measured response
    arma::vec yh; // estimated response
    arma::vec resid; // residuals

    arma::uword M; // number of measurements

    arma::vec alpha; // nonlinear parameter vector
    arma::vec beta; // linear parameter vector

    arma::mat Amat; // model matrix
    
    arma::umat jidx; // indexing matrix for sparse jacobian
    arma::mat mjac; // sparse matrix jacobian
    arma::mat J; // projected jacobian

    arma::mat dkc, dkrw; // cached terms used in varpro jacobian
private:
    std::shared_ptr<spdlog::logger> log;
    arma::uword feval, jeval; // evaluations of model function
};

class exp_model : public response_block
{
public:
    explicit exp_model(const arma::vec& m, const arma::vec& t);
    virtual ~exp_model();
    static constexpr auto name = "exp_model";

protected:
    const arma::vec tvec;

private:
    std::shared_ptr<spdlog::logger> log;
};
