#pragma once

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <armadillo>
#include "varpro_objects.h"

namespace py = pybind11;

typedef std::tuple<py::array, py::array> np_yJ;

const std::string hello();
py::buffer_info vec_buffer(arma::vec &m);
py::buffer_info mat_buffer(arma::mat &m);
void vec_np_init(arma::vec &v, py::array inp);
void mat_np_init(arma::mat &m, py::array inp);

//np_yJ package_yJ(const response_block&);
