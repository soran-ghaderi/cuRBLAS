add_library(cuRBLAS cuRBLAS.cpp)
target_include_directories(cuRBLAS PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
