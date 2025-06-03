# Create the cuRBLAS library with CUDA sources
add_library(cuRBLAS 
    cuRBLAS.cpp
    context.cu
)

# Set CUDA properties
set_target_properties(cuRBLAS PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include directories
target_include_directories(cuRBLAS PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link CUDA libraries
target_link_libraries(cuRBLAS PUBLIC 
    CUDA::cudart
    CUDA::curand
)
