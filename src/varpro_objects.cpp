#include <algorithm>
#include <exception>
#include <tuple>
#include <cmath>
#include <iostream>
#include "varpro_objects.h"

response_block::response_block(const arma::vec &m):
    y(m), 
    yh(m.n_elem), 
    resid(m.n_elem), 
    M(m.n_elem),
    feval(0),
    jeval(0),
    log(spdlog::get("varpro"))
{
    log->debug("in response_block::response_block()");
    log->debug("got vector with {} elements", M);
}

response_block::~response_block()
{
    log->debug("in response_block::~response_block()");
}

const yJ_pair response_block::get_yJ() const 
{
    return std::make_tuple(yh, J); 
}

void response_block::update_model(const arma::vec p, bool update_jac)
{
    using arma::mat;
    using arma::vec;

    log->debug("in response_block::update_model()");
    log->debug("current response vector: {}", p.t());
    alpha = p;

    log->debug("evaluating model");
    evaluate_model(p);
    ++feval;

    log->debug("calculating linear parameters");
    mat U, V;
    vec s;
    mat Ut, Vt;
    mat Apinv;
    mat Sinv;

    bool success = svd_econ(U, s, V, Amat);

    if(!success) {
        log->error("SVD decomposition failed");
        throw std::runtime_error("SVD decomposition failed");
    }

    Ut = U.t();
    Vt = V.t();
    log->debug("SVD sizes: U: {}, s: {}, V: {}",
            size(U), size(s), size(V));
    
    Sinv = arma::diagmat(1/s);
    Apinv = V*Sinv*Ut;
    beta = Apinv*y;
    yh = Amat*beta;
    resid = y - yh;
    log->debug("current beta: {}", beta.t());
    log->debug("Sizes: resid: {}, yh: {}, Apinv: {}, Sinv: {}", size(resid), size(yh), size(Apinv), size(Sinv));

    if(!update_jac)
        return;

    log->debug("evaluating model jacobian");
    evaluate_jacobian(p);
    ++jeval;

    log->debug("calculating the projected jacobian");
    dkc.set_size(M, jidx.n_cols);
    dkrw.set_size(Amat.n_cols, jidx.n_cols);
    log->debug("expected dkc size: {}, dkrw: {} ", size(dkc), size(dkrw));

    // unpack the dense jidx,mjac structure
    log->debug("updating dkc and dkrw");
    log->debug("jidx size: {}", size(jidx));
    arma::uword basis_no, param_no;
    for(auto i = 0; i < jidx.n_cols; i++) {
        basis_no = jidx(0, i);
        param_no = jidx(1, i);

        for(auto j = 0; j < M; j++) {
            //log->debug("Accessing (j={}, i={}", j, i);
            dkc(j, i) = mjac(j, i)*beta(basis_no);
            dkrw(basis_no, i) = mjac(j, i)*resid(j);
        }
    }

    log->debug("evaluating A and B");
    mat A = dkc - U*Ut*dkc;
    mat B = U*Sinv*Vt*dkrw;
    
    J = mat(M, alpha.n_elem, arma::fill::zeros); // fill jacobian with zeros
    log->debug("initialized J to size {}", size(J));

    log->debug("compressing A and B");
    for(auto i = 0; i < jidx.n_cols; i++) {
        basis_no = jidx(0, i);
        param_no = jidx(1, i);
        for(auto j = 0; j < M; ++j) 
            J(j, param_no) = -(A(j, i) + B(j, i)); 
    }
    log->debug("finished; counts: feval={}, jeval={}", feval, jeval);
}

exp_model::exp_model(const arma::vec& m, const arma::vec& t):
    response_block(m),
    tvec(t)
{
    log->debug("in exp_model::exp_model()");

    if(m.n_elem != t.n_elem)
        throw std::runtime_error("y and t vector lengths must match");

    Amat.set_size(M, 2);
    mjac.set_size(M, 1);
    jidx = arma::umat({{1, 0}}).t(); // jacobian only has one nonzero column
    log->debug("jidx initialized to \n{}", jidx);
}

exp_model::~exp_model()
{
    log->debug("in exp_model::~exp_model()");
}

void exp_model::evaluate_model(const arma::vec& p) 
{
    log->debug("in exp_model::evaluate_model()");
    double t;
    for(auto i = 0; i < M; ++i) {
        t = tvec(i);
        Amat(i, 0) = 1.;
        Amat(i, 1) = std::exp(-t*p(0));
    }
    log->debug("done updating Amat");
}

void exp_model::evaluate_jacobian(const arma::vec& p)
{
    log->debug("in exp_model::evaluate_jacobian()");
    double t;
    for(auto i = 0; i < M; ++i) {
        t = tvec(i);
        mjac(i, 0) = -t*std::exp(-t*p(0));
    }
    log->debug("done updating mjac");
}
