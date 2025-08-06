add_executable(tests
        tests.cu
)

target_link_libraries(tests PRIVATE 
    cuRBLAS 
    Catch2::Catch2WithMain
)

target_include_directories(tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include # For curblas/curblas.cuh and curblas/curblas.h
        ${CMAKE_CURRENT_SOURCE_DIR} # For tests.cu (if it needs local headers)
)

# Link CUDA libraries only if available
if(CUDA_FOUND)
    target_link_libraries(tests PRIVATE CUDA::cudart)
    set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

endif()

# Add test to CTest - simple and reliable approach
include(CTest)
add_test(NAME cuRBLAS_tests COMMAND tests)

# If Catch2 discovery is available, use it for better test reporting
if(COMMAND catch_discover_tests)
    catch_discover_tests(tests)
endif()
