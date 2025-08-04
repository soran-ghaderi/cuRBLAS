add_executable(cuRBLAS_app ReduceSum.cu)
target_link_libraries(cuRBLAS_app PRIVATE cuRBLAS)
