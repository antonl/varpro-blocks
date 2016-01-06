#pragma once

#include <armadillo>
#include <memory>
#include <array>
#include <vector>
#include <tuple>
#include "spdlog/spdlog.h"

typedef std::tuple<arma::uword, bool> dof_spec; // number of degrees of freedom, whether model includes intercept term 

struct fit_report
{
    double rse; // weighted residual squared error
    double R2; // coefficient of determination
    arma::uword mdof; // model degrees of freedom
    arma::uword ddof; // data degrees of freedom
    arma::vec se; // standard error of regression
    arma::mat cov; // variance/covariance matrix
    arma::mat cor; // correlation matrix
    arma::vec tratio; // Student's T ratio of parameters
    arma::vec parameters; // vector of parameter values
    std::vector<std::string> labels; // labels of parameter values
    arma::vec tresid; // Studentized residual

    fit_report(const arma::mat H,
               const arma::vec params, 
               const arma::vec residuals, 
               const dof_spec dof, 
               const std::vector<const char*> param_labels);
};

class response_block 
{
public:
    explicit response_block(const arma::vec& measured);
    virtual ~response_block();

    void update_model(const arma::vec p, bool update_jac=false);

    const std::tuple<arma::vec, arma::vec, arma::mat> get_yrJ() const;

    const std::tuple<arma::vec, arma::vec> get_params() const;
    const arma::vec get_target() const;
    const std::tuple<arma::mat, arma::umat, arma::mat, 
          arma::mat, arma::mat, arma::mat> get_internal() const;
    const std::tuple<arma::mat, arma::vec, arma::mat> get_svd() const;

    virtual const fit_report get_fit_report() const = 0;

    static constexpr auto name = "response_block";
    static constexpr dof_spec dof = std::make_tuple(0, true);
    static constexpr std::array<const char*, 1> param_labels = {"intercept"};

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
private:
};

class exp_model : public response_block
{
public:
    explicit exp_model(const arma::vec& m, const arma::vec& t);
    virtual ~exp_model();
    virtual const fit_report get_fit_report() const;

    static constexpr auto name = "exp_model";
    static constexpr dof_spec dof = std::make_tuple(2, true);
    static constexpr std::array<const char *, 3> param_labels = {"intercept", "A", "k1"};

protected:
    virtual void evaluate_model(const arma::vec&p);
    virtual void evaluate_jacobian(const arma::vec&p);
    const arma::vec tvec;

private:
};
