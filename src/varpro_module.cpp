#include <armadillo>
#include <string>
#include <tuple>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "varpro_objects.h"
#include "varpro_util.h"
#include "spdlog/spdlog.h"

namespace py  = pybind11;

PYBIND11_PLUGIN(varpro) {
    auto console = spdlog::stdout_logger_st("varpro");
    console->set_level(spdlog::level::info);
    console->debug("initializing module varpro");

    py::module m("varpro", "C++ implementation of multiresponse regression using variable projection");

    m.def("hello", &hello, "return a string containing a greeting");

    py::class_<fit_report>(m, "fit_report")
        .def_property_readonly("labels", [](const fit_report &m){return m.labels;})
        .def_property_readonly("parameters", 
                [](const fit_report &m){return m.parameters;})
        .def_property_readonly("dof", 
                [](const fit_report &m){return std::make_tuple(m.ddof, m.mdof);})
        .def_property_readonly("marginal_ci", 
                [](const fit_report &m){return m.marginal_ci;})
        .def_property_readonly("se", 
                [](const fit_report &m){return m.se;})
        .def_property_readonly("stats", 
                [](const fit_report &m){return std::make_tuple(m.chisqr, m.rms, m.rme);})
        .def_property_readonly("tresid", 
                [](const fit_report &m){return m.tresid;})
        .def_property_readonly("wresid", 
                [](const fit_report &m){return m.wresid;})
        .def_property_readonly("tstat", 
                [](const fit_report &m){return m.tstat;})
        .def_property_readonly("cor", 
                [](const fit_report &m){return m.cor;})
        .def("__repr__", 
                [](const fit_report &m, unsigned int width)
                {return m.printable_summary(width);}, py::arg("width") = 80);

    py::class_<response_block> rb(m, "_response_block");

    py::class_<exp_model>(m, exp_model::name, rb)
        .def(py::init<const arma::vec, const arma::vec>())
        .def_property_readonly("yrJ", [](const exp_model& m){return m.get_yrJ();})
        .def_property_readonly("fit", [](const exp_model& m){return m.get_params();})
        .def_property_readonly("target", [](const exp_model& m){return m.get_target();})
        .def_property_readonly("_internal", [](const exp_model& m){return m.get_internal();})
        .def_property_readonly("_svd", [](const exp_model& m){return m.get_svd();})
        .def("update_model", &exp_model::update_model, 
            "update the model", py::arg("p0"), py::arg("update_jac") = false)
        .def("fit_report", [](const exp_model& m, double alpha){return m.get_fit_report(alpha);}, py::arg("alpha") = 5.);

    py::module arma_mod = m.def_submodule("arma", "Python binding to armadillo types");
    py::class_<arma::vec>(arma_mod, "Vec")
        .def(py::init<const arma::uword>())
        .def("__init__", &vec_np_init)
        .def_property_readonly("shape", 
                [](const arma::vec &a){return std::make_tuple(a.n_rows);})
        .def_property_readonly("n_elem", [](const arma::vec &a){return a.n_elem;})
        .def("zeros", [](arma::vec &a){ a.zeros();})
        .def("ones", [](arma::vec &a){ a.ones();})
        .def("randn", [](arma::vec &a){ a.randn();})
        .def("print", [](const arma::vec&a){a.print();})
        .def("print", [](const arma::vec&a, std::string arg){a.print(arg);})
        .def_buffer(&vec_buffer);

    py::class_<arma::mat>(arma_mod, "Mat")
        .def(py::init<const arma::uword, const arma::uword>())
        .def("__init__", &mat_np_init)
        .def_property_readonly("shape", 
                [](const arma::mat &a){return std::make_tuple(a.n_rows, a.n_cols);})
        .def_property_readonly("n_elem", [](const arma::mat &a){return a.n_elem;})
        .def("zeros", [](arma::mat &a){ a.zeros();})
        .def("ones", [](arma::mat &a){ a.ones();})
        .def("randn", [](arma::mat &a){ a.randn();})
        .def("print", [](const arma::mat&a){a.print();})
        .def("print", [](const arma::mat&a, std::string arg){a.print(arg);})
        .def_buffer(&mat_buffer);

    py::class_<arma::umat>(arma_mod, "uMat")
        .def(py::init<const arma::uword, const arma::uword>())
        .def("__init__", &umat_np_init)
        .def_property_readonly("shape", 
                [](const arma::umat &a){return std::make_tuple(a.n_rows, a.n_cols);})
        .def_property_readonly("n_elem", [](const arma::umat &a){return a.n_elem;})
        .def("zeros", [](arma::umat &a){ a.zeros();})
        .def("ones", [](arma::umat &a){ a.ones();})
        .def("randn", [](arma::umat &a){ a.randn();})
        .def("print", [](const arma::umat&a){a.print();})
        .def("print", [](const arma::umat&a, std::string arg){a.print(arg);})
        .def_buffer(&umat_buffer);

    return m.ptr();
}
