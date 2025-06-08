add_executable(tests 
    tests.cpp 
)

target_link_libraries(tests PUBLIC 
    cuRBLAS 
    Catch2::Catch2WithMain
)

# Link CUDA libraries only if available
if(CUDA_FOUND)
    target_link_libraries(tests PUBLIC CUDA::cudart)
endif()

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(tests)
