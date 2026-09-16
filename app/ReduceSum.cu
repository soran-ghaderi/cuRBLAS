#include "curblas/curblas.cuh"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cooperative_groups.h>


int main() {
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, deviceId);
    if (!supportsCoopLaunch) {
        std::cerr << "Device does not support cooperative kernel launch" << std::endl;
        return 1;
    }

    int N = 1024;
    float h_input[N];

    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;
    float h_output;


    cudaMalloc((void**)&d_input, N * sizeof(float));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    cudaMalloc((void**)&d_output, blockSize * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, blockSize * sizeof(float));

    void *args[] = {&d_input, &d_output, &N};
    cudaLaunchCooperativeKernel((void*)curblas::reduceSum, dim3(numBlocks), dim3(blockSize), args, blockSize * sizeof(float));


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

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

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
