add_executable(tests
        tests.cu
)

target_link_libraries(tests PRIVATE 
    cuRBLAS 
    Catch2::Catch2WithMain
)

# Link CUDA libraries only if available
if(CUDA_FOUND)
    target_link_libraries(tests PRIVATE CUDA::cudart)
endif()

# Add test to CTest - simple and reliable approach
add_test(NAME cuRBLAS_tests COMMAND tests)

# If Catch2 discovery is available, use it for better test reporting
if(COMMAND catch_discover_tests)
    catch_discover_tests(tests)
endif()
