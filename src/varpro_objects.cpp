#include "varpro_objects.h"
#include "arma_objects.h"

cvarpro_block::cvarpro_block(const arma::vec measured):
    y(measured), 
    M(measured.n_elem), 
    yh(measured.n_elem), 
    resid(measured.n_elem),
    logger(spdlog::get("varpro.cvarpro_block"))
{
    logger->debug("in cvarpro_block::cvarpro_block(arma::vec)");
    logger->debug("got vector with {} measurements", M);
}

cvarpro_block::~cvarpro_block()
{
    logger->debug("in cvarpro_block::~cvarpro_block()");
}

void cvarpro_block::update_model(const arma::vec p, bool update_jac)
{
    logger->debug("in cvarpro_block::update_model()");
}

single_exp_block::single_exp_block(const arma::vec measured, const arma::vec t):
    cvarpro_block(measured),
    m_t(t)
{
    logger->debug("in single_exp_block::single_exp_block");

    if(measured.size() != t.size())
        throw std::logic_error("t and measured vectors must match size");

    P = 1;
    N = 2;
    Amat = arma::mat(M, N);
    jidx = {{0, 1}};
    mjac = arma::mat(M, 1);
}

single_exp_block::~single_exp_block()
{
    logger->debug("in single_exp_block::~single_exp_block");
}

void single_exp_block::_generate_model_matrix(const arma::vec p)
{
    logger->debug("in single_exp_block::_generate_model_matrix()");
    double t;
    for(arma::uword m=0; m < M; m++) {
        t = m_t(m);
        Amat(m, 0) = 1.;
        Amat(m, 1) = std::exp(-t*p(0));
    }
    logger->debug("done");
}

void single_exp_block::_generate_jacobian_matrix(const arma::vec p)
{
    logger->debug("in single_exp_block::_generate_jacobian_matrix()");
    double t;
    for(arma::uword m=0; m < M; m++) {
        t = m_t(m);
        mjac(m, 0) = -t*std::exp(-t*p(0));
    }
    logger->debug("done");
}

template <typename T> void varpro_block<T>::init_type()
{
    auto logger = spdlog::get("varpro");
    logger->debug("initialized varpro_block<{}>", T::name);

    super::behaviors().name(T::name);
    super::behaviors().doc("cvapro_block<T>");

    super::behaviors().readyType();
}

template <typename T> varpro_block<T>::varpro_block(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds):
    super::PythonClass(self, args, kwds),
    logger(spdlog::get("varpro.varpro_block")) 
{
    logger->debug("created varpro_block<{}>", T::name);
}

template <typename T> varpro_block<T>::~varpro_block()
{
    logger->debug("in varpro_block<{}> dtor", T::name);
}

arma::vec make_arma_vec(const Py::Object obj) 
{
    Py::Tuple args(obj);
    Py::Dict d;
    Py::Callable arma_vec_type(arma_vec::type());
    Py::PythonClassObject<arma_vec> v = arma_vec_type.apply(args, d);
    arma::vec tmp(*v.getCxxObject());
    return tmp;
}

template <> varpro_block<single_exp_block>::varpro_block(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds):
    super::PythonClass(self, args, kwds),
    m_block(make_arma_vec(args[0]), make_arma_vec(args[1]))
{
    logger->debug("in specialized varpro_block<{}> ctor", single_exp_block::name);
}
