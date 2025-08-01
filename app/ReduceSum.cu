#include "cuRBLAS/cuRBLAS.cuh"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>


int main() {
    // Example usage of the reduceSum kernel
    const int N = 1024;
    float *d_input, *d_output;
    float h_output;
    float h_input[N];

    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }



    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));


    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    cuRBLAS::reduceSum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    // Copy the result back to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Sum: " << h_output << std::endl;
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
