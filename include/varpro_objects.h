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
    double chisqr; // sum of squares of the residual
    double rms; // residual mean square (variance)
    double rme; // residual mean error (error)
    double rsqr; // coefficient of determination
    double alpha; // upper quantile of confidence
    double cond; // condition number of X vector
    arma::uword mdof; // model degrees of freedom
    arma::uword ddof; // data degrees of freedom
    arma::vec se; // standard error in parameters
    arma::mat cor; // correlation matrix
    arma::vec tstat; // Student's T statistic of parameters
    arma::vec parameters; // vector of parameter values
    std::vector<std::string> labels; // labels of parameter values
    std::vector<std::tuple<double, double, double>> marginal_ci;
    arma::vec wresid; // weighted residuals
    arma::vec tresid; // Studentized residual
    std::string model_name;

    fit_report(std::string model_name,
               arma::mat H,
               arma::vec params, 
               arma::vec residuals, 
               dof_spec dof, 
               std::vector<const char*> param_labels,
               double alpha);

    std::string printable_summary(unsigned int width = 80) const;
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

    virtual const fit_report get_fit_report(double alpha) const = 0;

    static const char *name;
    static const dof_spec dof;
    static const std::array<const char*, 1> param_labels;

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
    virtual const fit_report get_fit_report(double alpha = 5.) const;

    static const char *name;
    static const dof_spec dof;
    static const std::array<const char *, 3> param_labels;

protected:
    virtual void evaluate_model(const arma::vec&p);
    virtual void evaluate_jacobian(const arma::vec&p);
    const arma::vec tvec;

private:
};
