#include <armadillo>
#include <string>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "varpro_objects.h"

namespace py  = pybind11;

const std::string hello() {
    return "Hello, world!";
}

py::buffer_info vec_buffer(arma::vec &m) 
{
    py::buffer_info buf(
            m.memptr(), 
            sizeof(arma::vec::elem_type), 
            py::format_descriptor<arma::vec::elem_type>::value(),
            1,
            {m.n_rows,},
            {sizeof(arma::vec::elem_type),}
        );
    return buf;
}

py::buffer_info mat_buffer(arma::mat &m) 
{
    py::buffer_info buf(
            m.memptr(), 
            sizeof(arma::mat::elem_type), 
            py::format_descriptor<arma::mat::elem_type>::value(),
            2,
            {m.n_rows, m.n_cols},
            {sizeof(arma::mat::elem_type), sizeof(arma::vec::elem_type)*m.n_rows}
        );
    return buf;
}

void vec_np_init(arma::vec &v, py::array inp) 
{
    py::buffer_info info = inp.request();
    if(info.format != py::format_descriptor<arma::vec::elem_type>::value() || info.ndim != 1)
        throw std::runtime_error("incompatible buffer format");

    if(info.strides[0] == info.itemsize) {
        new (&v) arma::vec(reinterpret_cast<arma::vec::elem_type *>(info.ptr), 
            info.count);
    } else {
        throw std::runtime_error("array not contiguous");
    }
}

void mat_np_init(arma::mat &m, py::array inp) 
{
    py::buffer_info info = inp.request();
    if(info.format != py::format_descriptor<arma::mat::elem_type>::value() || info.ndim != 2)
        throw std::runtime_error("Incompatible buffer format!");

    if(info.strides[0] == info.itemsize && 
            info.strides[1] == (info.itemsize * info.shape[0])) {
        // F-contigious
        new (&m) arma::mat(reinterpret_cast<arma::mat::elem_type *>(info.ptr),
                info.shape[0], info.shape[1]);
    } else if (info.strides[1] == info.itemsize &&
            info.strides[0] == (info.itemsize * info.shape[1])) {
        // C-contigious
        new (&m) arma::mat(reinterpret_cast<arma::mat::elem_type *>(info.ptr),
                info.shape[1], info.shape[0]);
        arma::inplace_trans(m);
    } else {
        throw std::runtime_error("array not contiguous");
    }
}

PYBIND11_PLUGIN(varpro) {
    py::module m("varpro", "C++ implementation of multiresponse regression using variable projection");

    m.def("hello", &hello, "return a string containing a greeting");

    py::module arma_mod = m.def_submodule("arma", "Python binding to armadillo types");
    py::class_<arma::vec>(arma_mod, "Vec")
        .def(py::init<const arma::uword>())
        .def("__init__", &vec_np_init)
        .def_property_readonly("n_rows", [](const arma::vec &a){return a.n_rows;})
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
        .def_property_readonly("n_rows", [](const arma::mat &a){return a.n_rows;})
        .def_property_readonly("n_cols", [](const arma::mat &a){return a.n_cols;})
        .def_property_readonly("n_elem", [](const arma::mat &a){return a.n_elem;})
        .def("zeros", [](arma::mat &a){ a.zeros();})
        .def("ones", [](arma::mat &a){ a.ones();})
        .def("randn", [](arma::mat &a){ a.randn();})
        .def("print", [](const arma::mat&a){a.print();})
        .def("print", [](const arma::mat&a, std::string arg){a.print(arg);})
        .def_buffer(&mat_buffer);

    return m.ptr();
}
