This is an explanation of the file structure that the cookiecutter
generated for you:

* C++ source files:
  * `include/cuRBLAS/cuRBLAS.hpp` is the main
    C++ header that declares the interface of your library.
  * `src/cuRBLAS.cpp` is the main file that implements this library.
  * `app/cuRBLAS_app.cpp` is an executable that uses the library.
    This can e.g. be used to provide a command line interface for your project.
  * `tests/cuRBLAS_t.cpp` contains the unit tests for the library.
    The unit tests are written using Catch2. For further reading on what can be achieved
    with Catch2, we recommend [their tutorial](https://github.com/catchorg/Catch2/blob/devel/docs/tutorial.md).
  * `tests/tests.cpp` is the Catch2 testing driver. You do not need to change
    this. Placing this in a separate compilation unit than the unit test
    implementation decreases the compilation time of the test suite.
  * The `python/cuRBLAS` directory contains a Python
    package for the project. It contains a compiled Python module `_cuRBLAS`
    that CMake generates from the Pybind11 source file `python/cuRBLAS/_cuRBLAS.cpp`. Additionally, it contains pure Python sources (e.g. `__init__.py`) that
    allow to wrap Python functionality around the compiled module.
* CMake build system files
  * `CMakeLists.txt` describes the CMake configuration script. You can find such files
    in many directories. When CMake runs, the `CMakeLists.txt` from the top-level directory
    executes top to bottom. Whenever a command `add_subdirectory(<dir>)` is executed,
    the `CMakeLists.txt` file from the directory `<dir>` is immediately executed. A comprehensive
    reference of CMake's capabilities can be found in the [official CMake docs](https://cmake.org/documentation/).
    A well-written, opinionated book for beginners and experts is [Modern CMake](https://cliutils.gitlab.io/modern-cmake/).
  * `cuRBLASConfig.cmake.in` provides a template for the configuration
    installed alongside your project. This is required to implement the transitivity of your dependency
    on `CUDA`: If downstream projects use your library, they should
    automatically search for `CUDA`. The config file template implements
    exactly this logic.
* Documentation configuration files
  * The Doxygen documentation is configured directly from `doc/CMakeLists.txt`.
    To further configure the build, you can check the [Doxygen Configuration Manual](https://www.doxygen.nl/manual/config.html)
    for available options and add them with `set(DOXYGEN_<param> <value>)` before
    the call to `doxygen_add_docs`.
* Configuration for CI/Code analysis/Documentation services
  * `.github/workflows/ci.yml` describes the Github Workflow for Continuous
    integration. For further reading on workflow files, we recommend the
    [introduction into Github Actions](https://docs.github.com/en/free-pro-team@latest/actions/learn-github-actions/introduction-to-github-actions)
    and [the reference of available options](https://docs.github.com/en/free-pro-team@latest/actions/reference/workflow-syntax-for-github-actions).
  * `.gitlab-ci.yml` describes the configuration for Gitlab CI. For further
    reading, we recommend [Gitlabs quick start guide](https://docs.gitlab.com/ee/ci/quick_start/)
    and the [Gitlab CI configuration reference](https://docs.gitlab.com/ce/ci/yaml/)
  * `codecov.yml` configures the coverage checking from [codecov.io](https://codecov.io). The
    provided file is the default configuration plus suitable exclusions. For more options, check
    their [configuration reference](https://docs.codecov.io/docs/codecov-yaml).
* Markdown files with meta information on the project. [Markdown](https://www.markdownguide.org/basic-syntax/) is
  a good language for these files, as it is easy to write and rendered into something beautiful by your git repository
  hosting provider.
  * `README.md` is the file that users will typically see first when discovering your project.
  * `COPYING.md` provides a list of copyright holders.
  * `LICENSE.md` contains the license you selected.
  * `TODO.md` contains a list of TODOs after running the cookiecutter. Following the
    instructions in that file will give you a fully functional repository with a lot
    of integration into useful web services activated and running.
  * `FILESTRUCTURE.md` describes the generated files. Feel free to remove this from the
    repository if you do not need it.
* Other files
  * `.gitignore` contains a default selection of files to omit from version control.
  * `pyproject.toml` provides information about the Python project and its toolchain.
    `pip` uses this file to control the entire build process. Additionally, it can be
    used to configure tools like `pytest`.Additionally, this file is needed for the automated release process to PyPI.
  * `.github/workflows/pypi.yml` defines the workflow that deploys to PyPI. The
    configuration of this workflow is done in `pyproject.toml`.
  * `python/tests/test_python_bindings.py` defines a simple unit test suite for the Python
    bindings that is based on [Pytest](https://docs.pytest.org/en/stable/contents.html).
    `requirements-dev.txt` collects the required Python packages for running this
    test suite, they can be installed with `python -m pip install -r requirements-dev.txt`.
    The `pytest` run is configured in the `tools.pytest` section of `pyproject.toml`.
