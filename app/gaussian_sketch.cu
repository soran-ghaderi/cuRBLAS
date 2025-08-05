#include "curblas/curblas.cuh"
//#include "curblas/curblas.h"
//#include "curblas/curblas_types.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>



void printMatrix(const std::vector<float>& vec, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(10) << vec[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int rows = 4000;
    int cols = 5000;

    int totalElements = rows * cols;
    long long seed = 112345L;
    float scale = 1.0f;

    std::cout << "Generating a" << rows << " x " << cols << " gaussian sketch matrix." << std::endl;

    std::vector<float> h_sketch(totalElements);

    float* d_sketch;
    cudaMalloc((void**)&d_sketch, totalElements * sizeof(float));

    int blockSize = 256;

    // int totalElements = rows * cols;
    int elementsPerThread = 4; //test
    int totalThreads = (totalElements + elementsPerThread - 1) / elementsPerThread;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // int numBlocks = (totalElements + blockSize - 1) / blockSize;

    curblas::generateGaussianSketch<<<numBlocks, blockSize>>>(d_sketch, rows, cols, seed, scale);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_sketch);
        return -1;
    }


    cudaDeviceSynchronize();
    cudaStream_t stream;
    cudaStreamCreate(&stream);

//  bring the data back:
    cudaMemcpyAsync(h_sketch.data(), d_sketch, totalElements * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::cout << "result:" << rows << 'x' << cols << std::endl;

//    printMatrix(h_sketch, rows, cols);

    cudaFree(d_sketch);


}