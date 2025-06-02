#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuRBLAS/cuRBLAS.hpp"

namespace py = pybind11;

namespace cuRBLAS {

PYBIND11_MODULE(_cuRBLAS, m)
{
  m.doc() = "Python Bindings for cuRBLAS";
  m.def("add_one", &add_one, "Increments an integer value");
}

} // namespace cuRBLAS
