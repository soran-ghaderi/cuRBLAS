#include <catch2/catch_test_macros.hpp>
#include "cuRBLAS/cuRBLAS.hpp"
#include "cuRBLAS/curblas.h"
#include <cuda_runtime.h>
#include <cstring>

using namespace cuRBLAS;

TEST_CASE( "add_one", "[adder]" ){
  REQUIRE(add_one(0) == 1);
  REQUIRE(add_one(123) == 124);
  REQUIRE(add_one(-1) == 0);
}

TEST_CASE("cuRBLAS Context Management", "[context]") {
    
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
