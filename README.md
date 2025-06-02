# Welcome to cuRBLAS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/soran-ghaderi/cuRBLAS/ci.yml?branch=main)](https://github.com/soran-ghaderi/cuRBLAS/actions/workflows/ci.yml)
[![PyPI Release](https://img.shields.io/pypi/v/cuRBLAS.svg)](https://pypi.org/project/cuRBLAS)
[![codecov](https://codecov.io/gh/soran-ghaderi/cuRBLAS/branch/main/graph/badge.svg)](https://codecov.io/gh/soran-ghaderi/cuRBLAS)

# Prerequisites

Building cuRBLAS requires the following software installed:

* A C++17-compliant compiler
* CMake `>= 3.9`
* CUDA
* Doxygen (optional, documentation building is skipped if missing)
* The testing framework [Catch2](https://github.com/catchorg/Catch2) for building the test suite* Python `>= 3.8` for building Python bindings

# Building cuRBLAS

The following sequence of commands builds cuRBLAS.
It assumes that your current working directory is the top-level directory
of the freshly cloned repository:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

The build process can be customized with the following CMake variables,
which can be set by adding `-D<var>={ON, OFF}` to the `cmake` call:

* `BUILD_TESTING`: Enable building of the test suite (default: `ON`)
* `BUILD_DOCS`: Enable building the documentation (default: `ON`)
* `BUILD_PYTHON`: Enable building the Python bindings (default: `ON`)


If you wish to build and install the project as a Python project without
having access to C++ build artifacts like libraries and executables, you
can do so using `pip` from the root directory:

```
python -m pip install .
```

# Testing cuRBLAS

When built according to the above explanation (with `-DBUILD_TESTING=ON`),
the C++ test suite of `cuRBLAS` can be run using
`ctest` from the build directory:

```
cd build
ctest
```

The Python test suite can be run by first `pip`-installing the Python package
and then running `pytest` from the top-level directory:

```
python -m pip install .
pytest
```

# Documentation

cuRBLAS provides a Doxygen documentation. You can build
the documentation locally by making sure that `Doxygen` is installed on your system
and running this command from the top-level build directory:

```
cmake --build . --target doxygen
```

The web documentation can then be browsed by opening `doc/html/index.html` in your browser.
