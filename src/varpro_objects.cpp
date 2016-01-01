#include <algorithm>
#include <exception>
#include "varpro_objects.h"

response_block::response_block(const arma::vec &m):
    y(m), 
    yh(m.n_elem), 
    resid(m.n_elem), 
    M(m.n_elem),
    log(spdlog::get("varpro"))
{
    log->debug("in response_block::response_block()");
    log->debug("got vector with {} elements", M);
}

response_block::~response_block()
{
    log->debug("in response_block::~response_block()");
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
    log->debug("SVD sizes: U ({}), s ({}), V ({})",
            U.size(), s.size(), V.size());
    
    Sinv = arma::diagmat(1/s);
    Apinv = V*Sinv*Ut;
    beta = Apinv*y;
    yh = Amat*beta;
    resid = y - yh;
    log->debug("current beta: {}", beta.t());

    if(!update_jac)
        return;

    log->debug("evaluating model jacobian");
    evaluate_jacobian(p);
    ++jeval;

    log->debug("calculating the projected jacobian");
    dkc.set_size(M, jidx.n_cols);
    dkrw.set_size(Amat.n_cols, jidx.n_cols);

    // unpack the dense jidx,mjac structure
    arma::uword basis_no, param_no;
    for(auto i = 0; i < jidx.n_cols; ++i) {
        basis_no = jidx(i, 0);
        param_no = jidx(i, 1);

        for(auto j = 0; j < M; ++j) {
            dkc(j, i) = mjac(j, i)*beta(basis_no);
            dkrw(basis_no, i) = mjac(j, i)*resid(j);
        }
    }

    mat A = dkc - U*Ut*dkc;
    mat B = U*Sinv*Vt*dkrw;
    
    mat J(M, beta.n_elem);
    for(auto i = 0; i < jidx.n_cols; ++i) {
        basis_no = jidx(i, 0);
        param_no = jidx(i, 1);
        for(auto j = 0; j < M; ++j) 
            J(j, param_no) = -(A(j, i) + B(j, i)); 
    }
    log->debug("evaluated the projected jacobian");
}

exp_model::exp_model(const arma::vec& m, const arma::vec& t):
    response_block(m),
    tvec(t),
    log(spdlog::get("varpro"))
{
    log->debug("in exp_model::exp_model()");

    if(m.n_elem != t.n_elem)
        throw std::runtime_error("y and t vector lengths must match");
}

exp_model::~exp_model()
{
    log->debug("in exp_model::~exp_model()");
}
