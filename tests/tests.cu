#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "curblas/curblas.cuh"
#include "curblas/curblas.h"
#include <cuda_runtime.h>
#include <numeric>
#include <cstring>
#include <iostream>

//using namespace curblas;

void checkCudaError(cudaError err, const char* msg) {
    if (err != cudaSuccess) {
        FAIL(msg << ": " << cudaGetErrorString(err));
    }
}

//TEST_CASE( "add_one", "[adder]" ){
//  REQUIRE(add_one(0) == 1);
//  REQUIRE(add_one(123) == 124);
//  REQUIRE(add_one(-1) == 0);
//}
TEST_CASE("generateGaussianSketch kernel tests", "[gaussianSketch]") {

    SECTION("basic generation and distribution check") {
        int rows = 100;
        int cols = 64;

        int totalElements = rows * cols;
        long long seed = 42;
        float scale = 1.0f;

        float* d_sketch;
        checkCudaError(cudaMalloc((void**)&d_sketch, totalElements * sizeof(float)), "Failed to allocate device memory for sketch");

        // set up kernel
        int blockSize = 256;
        int elementPerThread = 4;
        int totalThreads = (totalElements + elementPerThread - 1) / elementPerThread;
        int numBlocks = (totalThreads + blockSize - 1) / blockSize;

        curblas::generateGaussianSketch<<<numBlocks, blockSize>>>(d_sketch, rows, cols, seed, scale);
        checkCudaError(cudaGetLastError(), "Kernel launch failed");
        checkCudaError(cudaDeviceSynchronize(), "Device synchronization failed");

        std::vector<float> h_sketch(totalElements);
        checkCudaError(cudaMemcpy(h_sketch.data(), d_sketch, totalElements * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy sketch data to host");

        // scale check

        double sum = std::accumulate(h_sketch.begin(), h_sketch.end(), 0.0f);
        double mean = sum / totalElements;

        double sqSum = 0.0;
        for (float val : h_sketch) {
            sqSum += (val - mean) * (val - mean);
        }

        double stdDev = std::sqrt(sqSum / totalElements);
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0.0, 0.1));
        REQUIRE_THAT(stdDev, Catch::Matchers::WithinAbs(scale, 0.1));

        checkCudaError(cudaFree(d_sketch), "Failed to free device memory for sketch");

    }
}
TEST_CASE("reduceSum kernel tests", "[reduceSum]") {
    SECTION("sum of N elements") {
        int N = 1024;
        std::vector<float> h_input(N, 1.0f);
        float *d_input, *d_output;

        checkCudaError(cudaMalloc((void**)&d_input, N * sizeof(float)), "Failed to allocate device memory for input");

        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        checkCudaError(cudaMalloc((void**)&d_output, blockSize * sizeof(float)), "Failed to allocate device memory for output");

        checkCudaError(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy input data to device");

        checkCudaError(cudaMemset(d_output, 0, blockSize * sizeof(float)), "Failed to initialize output memory");
//        curblas::reduceSum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);

        void *args[] = {&d_input, &d_output, &N};
//        curblas::reduceSum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
        cudaLaunchCooperativeKernel((void*)curblas::reduceSum, dim3(numBlocks), dim3(blockSize), args, blockSize * sizeof(float));


        checkCudaError(cudaGetLastError(), "Kernel launch failed");
//        checkCudaError(cudaDeviceSynchronize(), "Device synchronization failed");


    }
}

TEST_CASE("curblas Context Management", "[context]") {

    SECTION("Create and Destroy Handle") {
        curblasHandle_t handle = nullptr;

        // Test creation
        curblasStatus_t status = curblasCreate(&handle);
        REQUIRE(status == CURBLAS_STATUS_SUCCESS);
        REQUIRE(handle != nullptr);

        // Test destruction
        status = curblasDestroy(handle);
        REQUIRE(status == CURBLAS_STATUS_SUCCESS);
    }

    SECTION("Invalid Handle Creation") {
        curblasStatus_t status = curblasCreate(nullptr);
        REQUIRE(status == CURBLAS_STATUS_INVALID_VALUE);
    }

    SECTION("Version Information") {
        curblasHandle_t handle = nullptr;
        curblasCreate(&handle);

        int version = -1;
        curblasStatus_t status = curblasGetVersion(handle, &version);
        REQUIRE(status == CURBLAS_STATUS_SUCCESS);
        REQUIRE(version >= 0);

        curblasDestroy(handle);
    }

    SECTION("Stream Management") {
        curblasHandle_t handle = nullptr;
        curblasCreate(&handle);

        // Test getting default stream
        cudaStream_t stream = nullptr;
        curblasStatus_t status = curblasGetStream(handle, &stream);
        REQUIRE(status == CURBLAS_STATUS_SUCCESS);
        // Note: stream may be nullptr in CPU-only environments

        // Test setting custom stream (only if CUDA is available)
        int deviceCount = 0;
        cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
        if (cudaStatus == cudaSuccess && deviceCount > 0) {
            // CUDA is available, test stream operations
            cudaStream_t customStream;
            cudaStreamCreate(&customStream);

            status = curblasSetStream(handle, customStream);
            REQUIRE(status == CURBLAS_STATUS_SUCCESS);

            cudaStream_t retrievedStream;
            status = curblasGetStream(handle, &retrievedStream);
            REQUIRE(status == CURBLAS_STATUS_SUCCESS);
            REQUIRE(retrievedStream == customStream);

            cudaStreamDestroy(customStream);
        } else {
            // CPU-only environment, just test that set/get stream don't crash
            status = curblasSetStream(handle, nullptr);
            REQUIRE(status == CURBLAS_STATUS_SUCCESS);
        }

        curblasDestroy(handle);
    }

    SECTION("Accuracy Configuration") {
        curblasHandle_t handle = nullptr;
        curblasCreate(&handle);

        // Test setting accuracy
        curblasStatus_t status = curblasSetAccuracy(handle, CURBLAS_ACCURACY_HIGH);
        REQUIRE(status == CURBLAS_STATUS_SUCCESS);

        // Test getting accuracy
        curblasAccuracy_t accuracy;
        status = curblasGetAccuracy(handle, &accuracy);
        REQUIRE(status == CURBLAS_STATUS_SUCCESS);
        REQUIRE(accuracy == CURBLAS_ACCURACY_HIGH);

        curblasDestroy(handle);
    }

    SECTION("Sketch Type Configuration") {
        curblasHandle_t handle = nullptr;
        curblasCreate(&handle);

        // Test setting sketch type
        curblasStatus_t status = curblasSetSketchType(handle, CURBLAS_SKETCH_GAUSSIAN);
        REQUIRE(status == CURBLAS_STATUS_SUCCESS);

        curblasDestroy(handle);
    }

    SECTION("Random Seed Configuration") {
        curblasHandle_t handle = nullptr;
        curblasCreate(&handle);

        // Test setting random seed
        unsigned long long seed = 42ULL;
        curblasStatus_t status = curblasSetRandomSeed(handle, seed);
        REQUIRE(status == CURBLAS_STATUS_SUCCESS);

        curblasDestroy(handle);
    }

    SECTION("Math Mode Configuration") {
        curblasHandle_t handle = nullptr;
        curblasCreate(&handle);

        // Test setting math mode
        curblasStatus_t status = curblasSetMathMode(handle, CURBLAS_TENSOR_OP_MATH);
        REQUIRE(status == CURBLAS_STATUS_SUCCESS);

        curblasDestroy(handle);
    }

    SECTION("Status String Function") {
        const char* statusStr = curblasGetStatusString(CURBLAS_STATUS_SUCCESS);
        REQUIRE(statusStr != nullptr);
        REQUIRE(strlen(statusStr) > 0);

        statusStr = curblasGetStatusString(CURBLAS_STATUS_INVALID_VALUE);
        REQUIRE(statusStr != nullptr);
        REQUIRE(strlen(statusStr) > 0);
    }

    SECTION("Error Handling") {
        // Test invalid handle operations
        REQUIRE(curblasDestroy(nullptr) == CURBLAS_STATUS_INVALID_VALUE);
        REQUIRE(curblasGetVersion(nullptr, nullptr) == CURBLAS_STATUS_INVALID_VALUE);
        REQUIRE(curblasSetStream(nullptr, nullptr) == CURBLAS_STATUS_INVALID_VALUE);
        REQUIRE(curblasSetAccuracy(nullptr, CURBLAS_ACCURACY_MEDIUM) == CURBLAS_STATUS_INVALID_VALUE);
    }
}
