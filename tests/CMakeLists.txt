add_executable(tests 
    tests.cpp 
    cuRBLAS_t.cpp
    test_context.cpp
)

target_link_libraries(tests PUBLIC 
    cuRBLAS 
    Catch2::Catch2
    CUDA::cudart
)

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(tests)
