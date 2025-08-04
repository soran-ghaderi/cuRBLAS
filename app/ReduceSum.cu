#include "curblas/curblas.cuh"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cooperative_groups.h>


int main() {
    // Example usage of the reduceSum kernel
    int N = 1024;
    float h_input[N];

    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;
    float h_output;


    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(float));
//    cudaMalloc((void**)&d_output, sizeof(float));


    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    cudaMalloc((void**)&d_output, blockSize * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, blockSize * sizeof(float));

//    curblas::reduceSum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    void *args[] = {&d_input, &d_output, &N};
    cudaLaunchCooperativeKernel((void*)curblas::reduceSum, dim3(numBlocks), dim3(blockSize), args, blockSize * sizeof(float));


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy the result back to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);


    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final Sum (from GPU reduction with Cooperative Groups): " << h_output << std::endl;


    std::vector<float> h_partialSums(numBlocks);
    cudaMemcpy(h_partialSums.data(), d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float finalSum = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        finalSum += h_partialSums[i];
        std::cout << "Partial sum from block " << i << ": " << h_partialSums[i] << std::endl;
    }


    std::cout << "Sum: " << finalSum << std::endl;
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
