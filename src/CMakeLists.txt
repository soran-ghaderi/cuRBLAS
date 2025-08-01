# Create the cuRBLAS library
if(CUDA_FOUND)

    # Create the cuRBLAS library with CUDA sources
    add_library(cuRBLAS
            cuRBLAS.cu
        context.cu
    )
    
    # Set CUDA properties
    set_target_properties(cuRBLAS PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Link CUDA libraries
    target_link_libraries(cuRBLAS PUBLIC 
        CUDA::cudart
    )
    
    # Define CUDA availability for code
    target_compile_definitions(cuRBLAS PUBLIC CURBLAS_CUDA_AVAILABLE)
    
    message(STATUS "Building cuRBLAS with CUDA support")

    # Include directories
    target_include_directories(cuRBLAS PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

else()
    # Create CPU-only version
    add_library(cuRBLAS
            cuRBLAS.cu
    )
    
    message(STATUS "Building cuRBLAS without CUDA support")
endif()


message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

#include_directories(/usr/local/cuda/include)
