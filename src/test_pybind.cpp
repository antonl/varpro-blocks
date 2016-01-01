#include "pybind11/pybind11.h"
#include <armadillo>

namespace py = pybind11;

int add(int i, int j) {
    return i+j;
}

PYBIND11_PLUGIN(test_pybind) {
    py::module m("test_pybind", "a quick test module");

    m.def("add", &add, "a function that adds two numbers");

    return m.ptr();
}
