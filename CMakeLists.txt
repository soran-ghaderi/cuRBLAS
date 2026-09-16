cmake_minimum_required(VERSION 3.18)

project(cuRBLAS VERSION 0.0.1 LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # has to be answered before enable_language(CUDA) caches a default of its own
    if(NOT DEFINED CURBLAS_ARCHS_AUTO)
        if(DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CURBLAS_ARCHS_AUTO OFF CACHE INTERNAL "architectures came from the command line")
        else()
            set(CURBLAS_ARCHS_AUTO ON CACHE INTERNAL "architectures are chosen by cuRBLAS")
        endif()
    endif()

    enable_language(CUDA)
    set(CUDA_FOUND TRUE)

    # gate each tier on the toolkit that can build it, else nvcc targets only its own
    # default arch and the binary won't launch on a newer gpu
    if(CURBLAS_ARCHS_AUTO)
        set(CURBLAS_CUDA_ARCHS "")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
            list(APPEND CURBLAS_CUDA_ARCHS 60 61 70)    # pascal, volta: dropped in cuda 13
        endif()
        list(APPEND CURBLAS_CUDA_ARCHS 75 80 86)        # turing, ampere
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
            list(APPEND CURBLAS_CUDA_ARCHS 89)          # ada
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
            list(APPEND CURBLAS_CUDA_ARCHS 90)          # hopper
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
            list(APPEND CURBLAS_CUDA_ARCHS 100 120)     # blackwell
        endif()
        # force, a plain set() loses to the default enable_language already cached
        # no "-virtual" entry, nvlink strips the ptx back out of the linked target
        set(CMAKE_CUDA_ARCHITECTURES "${CURBLAS_CUDA_ARCHS}" CACHE STRING "CUDA architectures" FORCE)
    endif()
    message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION}, targeting architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
else()
    set(CUDA_FOUND FALSE)
    message(WARNING "CUDA not found - building CPU-only version")
endif()

cmake_policy(SET CMP0074 NEW)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# msvc won't link if runtime libs or iterator debug levels disagree across targets
if(WIN32 AND MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd /D_ITERATOR_DEBUG_LEVEL=2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /D_ITERATOR_DEBUG_LEVEL=0")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MD /D_ITERATOR_DEBUG_LEVEL=0")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /MD /D_ITERATOR_DEBUG_LEVEL=0")
endif()

option(BUILD_PYTHON "Enable building of Python bindings" OFF)
option(BUILD_DOCS "Enable building of documentation" ON)

if(CUDA_FOUND)
    find_package(CUDAToolkit REQUIRED)
endif()
find_package(Threads REQUIRED)

add_subdirectory(src)
add_subdirectory(app)

include(CTest)
if(BUILD_TESTING)
  find_package(Catch2 3 REQUIRED)

  if(TARGET Catch2::Catch2WithMain)
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
  add_subdirectory(docs)
endif()

if(BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
  pybind11_add_module(_cuRBLAS python/curblas/_curblas.cpp)
  target_link_libraries(_cuRBLAS PUBLIC cuRBLAS)
  install(TARGETS _cuRBLAS DESTINATION .)
endif()

add_library(cuRBLAS::cuRBLAS ALIAS cuRBLAS)

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

include(FeatureSummary)
feature_summary(WHAT ALL)
