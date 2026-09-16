#include "curblas/curblas.cuh"
#include "curblas/curblas.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
    int rows = 4000;
    int cols = 5000;
    int totalElements = rows * cols;
    float scale = 1.0f;

    std::cout << "Generating a " << rows << " x " << cols << " gaussian sketch matrix." << std::endl;

    curblasHandle_t handle = nullptr;
    if (curblasCreate(&handle) != CURBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create curblas handle" << std::endl;
        return -1;
    }

    int blockSize = 256;
    int elementsPerThread = 4;
    int totalThreads = (totalElements + elementsPerThread - 1) / elementsPerThread;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    if (curblasEnsureRngCapacity(handle, totalThreads) != CURBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to allocate RNG states" << std::endl;
        curblasDestroy(handle);
        return -1;
    }

    std::vector<float> h_sketch(totalElements);
    float* d_sketch;
    cudaMalloc((void**)&d_sketch, totalElements * sizeof(float));

    curandStatePhilox4_32_10_t* states = curblasGetRngStates(handle);
    curblas::gaussianSketchOperator<<<numBlocks, blockSize>>>(states, d_sketch, rows, cols, scale);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_sketch);
        curblasDestroy(handle);
        return -1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_sketch.data(), d_sketch, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "result: " << rows << 'x' << cols << std::endl;

    cudaFree(d_sketch);
    curblasDestroy(handle);
}