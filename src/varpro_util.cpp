#include <armadillo>
#include <string>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "varpro_objects.h"

namespace py  = pybind11;

typedef std::tuple<py::array, py::array> np_yJ;

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

/*
np_yJ package_yJ(const response_block &b) 
{
    auto x = b.get_yJ();
    return std::make_tuple(
            py::array(vec_buffer(std::get<0>(x))),
            py::array(mat_buffer(std::get<1>(x)))
            ); 
}
*/
