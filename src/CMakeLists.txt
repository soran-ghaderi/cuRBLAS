if(CUDA_FOUND)
    add_library(cuRBLAS
        curblas.cu
        context.cu
        rgemm.cu
    )

    set_target_properties(cuRBLAS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_link_libraries(cuRBLAS PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )

    target_compile_definitions(cuRBLAS PUBLIC CURBLAS_CUDA_AVAILABLE)

    target_include_directories(cuRBLAS PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    message(STATUS "Building cuRBLAS with CUDA support")
else()
    add_library(cuRBLAS
        curblas.cu
    )

    message(STATUS "Building cuRBLAS without CUDA support")
endif()
