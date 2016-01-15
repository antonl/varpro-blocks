#include <algorithm>
#include <exception>
#include <tuple>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <iterator>
#include "boost/math/distributions.hpp"
#include "varpro_objects.h"

fit_report::fit_report(
        std::string name,
        arma::mat H, 
        arma::vec params, 
        arma::vec residuals, 
        dof_spec dof, 
        std::vector<const char*> param_labels,
        double alpha):
    parameters(params),
    wresid(residuals),
    model_name(name),
    alpha(alpha)
{
    // copy over labels 
    std::for_each(param_labels.begin(), param_labels.end(), 
            [&](const char *s){labels.push_back(s);});

    mdof = std::get<0>(dof);
    ddof = wresid.n_elem - mdof - (std::get<1>(dof)? 1 : 0);
    chisqr = arma::sum(wresid.t()*wresid);
    rms = chisqr/ddof;
    rme = std::sqrt(rms);

    // QR decompose H to calculate correlation matrix
    arma::mat Q, R, Rinv, I;
    arma::qr_econ(Q, R, H);
    I = arma::eye(R.n_cols, R.n_cols);
    arma::solve(Rinv, arma::trimatu(R), I);
    cond = arma::cond(H);

    arma::mat L(arma::size(Rinv));
    arma::vec ln(Rinv.n_rows);

    for(auto i = 0; i < Rinv.n_rows; i++) {
        ln(i) = arma::norm(Rinv.row(i));
        L.row(i) = Rinv.row(i)/ln(i);
    }

    cor = L*L.t();
    se = rme*ln;

    if(alpha <= 0.) throw std::logic_error("alpha parameter must be positive");

    //double tval = gsl_cdf_tdist_Qinv(alpha/200., ddof); 
    using boost::math::students_t;
    using boost::math::complement;
    using boost::math::quantile;
    students_t tgen(ddof);
	double tval = quantile(complement(tgen, alpha / 200));

    double param;
    for(auto i = 0; i < parameters.n_elem; i++) {
        param = parameters(i);
        marginal_ci.push_back(
                std::make_tuple(param, param - rme*tval, param + rme*tval)
        );
    }

    tstat = parameters/se;

    double st;
    tresid.copy_size(wresid);
    for(auto i = 0; i < Q.n_rows; i++) {
        st = std::sqrt(1. - arma::norm(Q.row(i)));
        tresid(i) = wresid(i)*rme*st;
    }
}

// http://stackoverflow.com/questions/14861018/center-text-in-fixed-width-field-with-stream-manipulators-in-c
template<typename charT, typename traits = std::char_traits<charT> >
class center_helper {
    std::basic_string<charT, traits> str_;
public:
    center_helper(std::basic_string<charT, traits> str) : str_(str) {}
    template<typename a, typename b>
    friend std::basic_ostream<a, b>& operator<<(std::basic_ostream<a, b>& s, const center_helper<a, b>& c);
};

template<typename charT, typename traits = std::char_traits<charT> >
center_helper<charT, traits> centered(std::basic_string<charT, traits> str) {
    return center_helper<charT, traits>(str);
}

center_helper<std::string::value_type, std::string::traits_type> centered(const std::string& str) {
    return center_helper<std::string::value_type, std::string::traits_type>(str);
}

template<typename charT, typename traits>
std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& s, const center_helper<charT, traits>& c) {
    std::streamsize w = s.width();
    if (w > c.str_.length()) {
        std::streamsize left = (w + c.str_.length()) / 2;
        s.width(left);
        s << c.str_;
        s.width(w - left);
        s << "";
    } else {
        s << c.str_;
    }
    return s;
}
std::string fit_report::printable_summary(unsigned int width) const
{
    using namespace std;
    stringstream s;

    s << setw(width) << centered("Fit Summary") << endl;
    s << setw(width) << setfill('=') << " " << endl;
    s << setfill(' ') 
      << setw(int(width/4)) << left << "Model" 
      << setw(int(width/4) - 2) << right << model_name
      << "    "
      << setw(int(width/4) - 2) << left << "R-squared"
      << setw(int(width/4) - 1) << right << "***" << " " << endl;

    s << setw(int(width/4)) << left << "No. observations" 
      << setw(int(width/4) - 2) << right << wresid.n_elem
      << "    "
      << setw(int(width/4) - 2) << left << "Adj. R-squared"
      << setw(int(width/4) - 1) << right << "***" << " " << endl;
        
    s << setw(int(width/4)) << left << "Df residuals" 
      << setw(int(width/4) - 2) << right << ddof
      << "    "
      << setw(int(width/4) - 2) << left << "F-statistic"
      << setw(int(width/4) - 1) << right << "***" << " " << endl;

    s << setw(int(width/4)) << left << "Df model" 
      << setw(int(width/4) - 2) << right << mdof
      << "    "
      << setw(int(width/4) - 2) << left << "Prob. F-statistic"
      << setw(int(width/4) - 1) << right << "***" << " " << endl;

    s << setw(width) << setfill('=')  << " " << setfill(' ')<< endl;

    stringstream conf_field;
    conf_field << "[" << setprecision(3) << (1 - alpha/100.) 
        << "% Conf. Int.]";

    s << setw(int(width/7)) << left << " " 
      << setw(int(width/7)) << right << "coef"
      << setw(int(width/7)) << right << "std err"
      << setw(int(width/7)) << right << "t-stat"
      << setw(int(width/7)) << right << "p value"
      << setw(width - 5*int(width/7) - 1)  << right
      << conf_field.str() << " " << endl;

    s << setw(width) << setfill('-')  << " " << setfill(' ')<< endl;
    s << setprecision(2) << scientific;

    for(auto i = 0; i < parameters.n_elem; i++) {
        auto ci = marginal_ci.at(i);
        s << setw(int(width/7)) << left << labels.at(i)
          << setw(int(width/7)) << right << get<0>(ci)
          << setw(int(width/7)) << right << se(i)
          << setw(int(width/7)) << right << tstat(i)
          << setw(int(width/7)) << right << "***"
          << setw(int(width/7)) << right << get<1>(ci)
          << setw(width - 6*int(width/7) - 1) << right << get<2>(ci) << " " << endl;
    }
    s << setw(width) << setfill('=') << " " << endl;

    s << setfill(' ') 
      << setw(int(width/4)) << left << "Cond. No." 
      << setw(int(width/4) - 2) << right << cond
      << "    "
      << setw(int(width/4) - 2) << left << " "
      << setw(int(width/4) - 1) << right << " " << " " << endl;

    s << setw(width) << setfill('=') << " " << endl;

    return s.str();
}

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

const char *response_block::name = "response_block";
const dof_spec response_block::dof = std::make_tuple(0, true);
const std::array<const char *, 1> response_block::param_labels = {"intercept"};

const std::tuple<arma::vec, arma::vec, arma::mat> response_block::get_yrJ() const
{
    return std::make_tuple(yh, resid, J); 
}

const arma::vec response_block::get_target() const
{
    return y;
}

const std::tuple<arma::vec, arma::vec> response_block::get_params() const 
{
    return std::make_tuple(alpha, beta);
}

const std::tuple<arma::mat, arma::vec, arma::mat> response_block::get_svd() const 
{
    return std::make_tuple(U, s, V);
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
    //mat U, V;
    //vec s;
    mat Apinv;
    mat Sinv;

    bool success = svd_econ(U, s, V, Amat);

    if(!success) {
        log->error("SVD decomposition failed");
        throw std::runtime_error("SVD decomposition failed");
    }

    mat Ut = U.t();
    mat Vt = V.t();
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
    dkc.zeros();
    dkrw.zeros();
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
            dkc(j, i) += mjac(j, i)*beta(basis_no);
            dkrw(basis_no, i) += mjac(j, i)*resid(j);
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
        for(auto j = 0; j < M; j++) 
            J(j, param_no) += (A(j, i) + B(j, i)); // removed minus sign for LM method
            //J(j, param_no) += -A(j, i); 
            //J(j, param_no) += -B(j, i); 
    }
    log->debug("finished; counts: feval={}, jeval={}", feval, jeval);
}

const std::tuple<arma::mat, arma::umat, arma::mat, 
      arma::mat, arma::mat, arma::mat> response_block::get_internal() const
{
    return std::make_tuple(Amat, jidx, mjac, dkc, dkrw, J);
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

const char *exp_model::name = "exp_model";
const dof_spec exp_model::dof = std::make_tuple(2, true);
const std::array<const char *, 3> exp_model::param_labels = {"intercept", "A", "k1" };

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

const fit_report exp_model::get_fit_report(double _a) const
{
    log->debug("generating fit_report");
    // generate H matrix
    arma::mat H;
    H.set_size(M, Amat.n_cols + J.n_cols);

    // copy over both the linear jacobian (which is just the model matrix)
    // and the projected jacobian
    std::copy(Amat.begin(), Amat.end(), H.begin());
    std::copy(J.begin(), J.end(), H.begin_col(Amat.n_cols));

    arma::vec params;
    params.set_size(beta.n_elem + alpha.n_elem);
    // Do the same with the parameters
    std::copy(beta.cbegin(), beta.cend(), params.begin());
    std::copy(alpha.cbegin(), alpha.cend(), params.begin_row(beta.n_elem));

    std::vector <const char *> labels;
    std::for_each(param_labels.begin(), param_labels.end(), 
            [&](const char *s){labels.push_back(s);});
    log->debug("vector size: {}", labels.size());
    log->debug("alpha parameter: {}", _a);

    return fit_report(name, H, params, resid, dof, labels, _a);
}
