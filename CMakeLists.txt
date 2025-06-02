cmake_minimum_required(VERSION 3.9)

# Set a name and a version number for your project:
project(cuRBLAS VERSION 0.0.1 LANGUAGES CXX)

# Set CMake policies for this project

# We allow <Package>_ROOT (env) variables for locating dependencies
cmake_policy(SET CMP0074 NEW)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compilation options
option(BUILD_PYTHON "Enable building of Python bindings" OFF)
option(BUILD_DOCS "Enable building of documentation" ON)

# Find external dependencies
find_package(CUDA)

# compile the library
add_subdirectory(src)


# compile the application
add_subdirectory(app)

# compile the tests
include(CTest)
if(BUILD_TESTING)
  find_package(Catch2 REQUIRED)
  include(Catch)
  add_subdirectory(tests)
endif()

if(BUILD_DOCS)
  # Add the documentation
  add_subdirectory(doc)
endif()
if(BUILD_PYTHON)
  # Add Python bindings
  find_package(pybind11 REQUIRED)
  # Compile the Pybind11 module
  pybind11_add_module(_cuRBLAS python/cuRBLAS/_cuRBLAS.cpp)
  target_link_libraries(_cuRBLAS PUBLIC cuRBLAS)

  # Install the Python module shared library
  install(TARGETS _cuRBLAS DESTINATION .)
endif()

# Add an alias target for use if this project is included as a subproject in another project
add_library(cuRBLAS::cuRBLAS ALIAS cuRBLAS)

# Install targets and configuration
install(
  TARGETS cuRBLAS
  EXPORT cuRBLAS-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
  EXPORT cuRBLAS-targets
  FILE cuRBLASTargets.cmake
  NAMESPACE cuRBLAS::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuRBLAS)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cuRBLASConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cuRBLASConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuRBLAS)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/cuRBLASConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuRBLAS)

export(
  EXPORT cuRBLAS-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/cuRBLASTargets.cmake
  NAMESPACE cuRBLAS::)

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
