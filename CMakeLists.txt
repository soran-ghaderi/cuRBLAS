cmake_minimum_required(VERSION 3.18)

# Set a name and a version number for your project:
# Start with just CXX, add CUDA if available
project(cuRBLAS VERSION 0.0.1 LANGUAGES CXX)

# Check if CUDA is available and enable it if found
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_FOUND TRUE)
    message(STATUS "CUDA found and enabled")
    
    # Set CUDA architectures - this fixes the CMAKE_CUDA_ARCHITECTURES error
    # Supporting common GPU architectures: Pascal, Volta, Turing, Ampere, Ada Lovelace
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90" CACHE STRING "CUDA architectures")
    
    # Define CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # CUDA specific settings
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
else()
    set(CUDA_FOUND FALSE)
    message(WARNING "CUDA not found - building CPU-only version")
endif()

# Set CMake policies for this project

# We allow <Package>_ROOT (env) variables for locating dependencies
cmake_policy(SET CMP0074 NEW)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Windows-specific settings to prevent runtime library mismatches
if(WIN32 AND MSVC)
    # Ensure consistent runtime library settings across all targets
    # Use /MDd for Debug builds and /MD for Release builds
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MD")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /MD")
    
    # Ensure consistent iterator debug levels
    # Use level 2 for Debug builds and level 0 for Release builds
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_ITERATOR_DEBUG_LEVEL=0")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /D_ITERATOR_DEBUG_LEVEL=0")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /D_ITERATOR_DEBUG_LEVEL=0")
    
    message(STATUS "Windows MSVC runtime library flags configured for consistent linking")
endif()

# Compilation options
option(BUILD_PYTHON "Enable building of Python bindings" OFF)
option(BUILD_DOCS "Enable building of documentation" ON)

# Find external dependencies
if(CUDA_FOUND)
    find_package(CUDAToolkit REQUIRED)
endif()
find_package(Threads REQUIRED)

# compile the library
add_subdirectory(src)


# compile the application
add_subdirectory(app)

# compile the tests
include(CTest)
if(BUILD_TESTING)
  find_package(Catch2 3 REQUIRED)
  
  # Try to include Catch2 discovery functions if available
  if(TARGET Catch2::Catch2WithMain)
    # Look for the Catch2 CMake module in common locations
    find_file(CATCH2_CMAKE_MODULE 
      NAMES Catch2.cmake 
      PATHS 
        ${Catch2_DIR}
        ${Catch2_DIR}/../../../lib/cmake/Catch2
        /usr/local/lib/cmake/Catch2
        /usr/lib/cmake/Catch2
      NO_DEFAULT_PATH
    )
    
    if(CATCH2_CMAKE_MODULE)
      include(${CATCH2_CMAKE_MODULE})
    else()
      message(STATUS "Catch2 discovery module not found, using basic test registration")
    endif()
  endif()
  
  add_subdirectory(tests)
endif()

if(BUILD_DOCS)
  # Add the documentation
  add_subdirectory(docs)
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

#include_directories(${CUDA_INCLUDE_DIRS})
#target_include_directories(cuRBLAS
#        PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}/include
#        ${CUDA_INCLUDE_DIRS}
#)
#target_link_libraries(cuRBLAS
#        PUBLIC
#        ${CUDA_LIBRARIES}
#)
#include_directories(/usr/local/cuda/include)
# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
