#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "curblas/curblas.hpp"

namespace py = pybind11;

namespace cuRBLAS {

PYBIND11_MODULE(_curblas, m)
{
  m.doc() = "Python Bindings for curblas";
//  m.def("add_one", &add_one, "Increments an integer value");
}

} // namespace curblas
