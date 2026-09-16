add_executable(tests
        tests.cu
)

target_link_libraries(tests PRIVATE
    cuRBLAS
    Catch2::Catch2WithMain
)

target_include_directories(tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CUDA_FOUND)
    target_link_libraries(tests PRIVATE CUDA::cudart)
    set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

include(CTest)
add_test(NAME cuRBLAS_tests COMMAND tests)

if(COMMAND catch_discover_tests)
    catch_discover_tests(tests)
endif()
