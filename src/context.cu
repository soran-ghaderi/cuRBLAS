#include "curblas/curblas.h"
#include "curblas/curblas_types.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

#include <memory>
#include <cmath>

// Define M_PI if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ __launch_bounds__(256, 4)
void initRngStates(curandStatePhilox4_32_10_t* states, size_t numStates, unsigned long long seed) {
    size_t gridStride = blockDim.x * gridDim.x;
    for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < numStates; id += gridStride) {
        curand_init(seed, id, 0, &states[id]);
    }
}

struct curblasContext {
    cudaStream_t stream;
    bool ownsStream;
    
    cublasHandle_t cublasHandle;
    
    unsigned long long seed;
    
    curandStatePhilox4_32_10_t* rngStates;
    size_t rngStatesCapacity;
    
    void* workspace;
    size_t workspaceSize;
    
    curblasAccuracy_t accuracy;
    curblasSketchType_t defaultSketchType;
    curblasMath_t mathMode;
    
    // Device information
    int deviceId;
    
    // Version information
    int version;
    
    // Constructor
    curblasContext() : 
        stream(nullptr),
        ownsStream(false),
        cublasHandle(nullptr),
        seed(1234ULL),
        rngStates(nullptr),
        rngStatesCapacity(0),
        workspace(nullptr),
        workspaceSize(0),
        accuracy(CURBLAS_ACCURACY_MEDIUM),
        defaultSketchType(CURBLAS_SKETCH_AUTO),
        mathMode(CURBLAS_DEFAULT_MATH),
        deviceId(-1),
        version((CURBLAS_VERSION_MAJOR << 16) | (CURBLAS_VERSION_MINOR << 8) | CURBLAS_VERSION_PATCH)
    {}
};

/*
 * ============================================================================
 * curblas Context Management Implementation
 * ============================================================================
 */

curblasStatus_t curblasCreate(curblasHandle_t* handle) {
    if (handle == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    try {
        // Allocate context
        curblasContext* ctx = new curblasContext();
        
        // Check if CUDA is available and there are devices
        int deviceCount = 0;
        cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
        if (cudaStatus != cudaSuccess || deviceCount == 0) {
            // CUDA not available or no devices - create a CPU-only context
            ctx->deviceId = -1;
            ctx->stream = nullptr;
            ctx->ownsStream = false;
            *handle = ctx;
            return CURBLAS_STATUS_SUCCESS;
        }
        
        // Get current device
        cudaStatus = cudaGetDevice(&ctx->deviceId);
        if (cudaStatus != cudaSuccess) {
            delete ctx;
            return CURBLAS_STATUS_NOT_INITIALIZED;
        }
        
        // Create default stream
        cudaStatus = cudaStreamCreate(&ctx->stream);
        if (cudaStatus != cudaSuccess) {
            delete ctx;
            return CURBLAS_STATUS_ALLOC_FAILED;
        }
        ctx->ownsStream = true;

        cublasStatus_t cublasStatus = cublasCreate(&ctx->cublasHandle);
        if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
            cudaStreamDestroy(ctx->stream);
            delete ctx;
            return CURBLAS_STATUS_ALLOC_FAILED;
        }
        cublasSetStream(ctx->cublasHandle, ctx->stream);

        *handle = ctx;
        return CURBLAS_STATUS_SUCCESS;
        
    } catch (const std::bad_alloc&) {
        return CURBLAS_STATUS_ALLOC_FAILED;
    } catch (...) {
        return CURBLAS_STATUS_INTERNAL_ERROR;
    }
}

curblasStatus_t curblasDestroy(curblasHandle_t handle) {
    if (handle == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    
    if (ctx->workspace) {
        cudaFree(ctx->workspace);
    }
    
    if (ctx->rngStates) {
        cudaFree(ctx->rngStates);
    }
    
    if (ctx->cublasHandle) {
        cublasDestroy(ctx->cublasHandle);
    }
    
    if (ctx->stream && ctx->ownsStream) {
        cudaStreamDestroy(ctx->stream);
    }
    
    delete ctx;
    
    return CURBLAS_STATUS_SUCCESS;
}

curblasStatus_t curblasGetVersion(curblasHandle_t handle, int* version) {
    if (handle == nullptr || version == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    *version = ctx->version;
    
    return CURBLAS_STATUS_SUCCESS;
}

curblasStatus_t curblasSetStream(curblasHandle_t handle, cudaStream_t streamId) {
    if (handle == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    
    // For CPU-only contexts, just return success (no CUDA operations)
    if (ctx->deviceId == -1) {
        return CURBLAS_STATUS_SUCCESS;
    }
    
    // Destroy old stream if we own it
    if (ctx->stream && ctx->ownsStream) {
        cudaStreamDestroy(ctx->stream);
    }
    
    ctx->stream = streamId;
    ctx->ownsStream = false;
    
    if (ctx->cublasHandle) {
        cublasSetStream(ctx->cublasHandle, streamId);
    }
    
    return CURBLAS_STATUS_SUCCESS;
}

curblasStatus_t curblasGetStream(curblasHandle_t handle, cudaStream_t* streamId) {
    if (handle == nullptr || streamId == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    
    if (ctx->deviceId == -1) {
        *streamId = nullptr;
        return CURBLAS_STATUS_SUCCESS;
    }
    
    *streamId = ctx->stream;
    
    return CURBLAS_STATUS_SUCCESS;
}

/*
 * ============================================================================
 * curblas Internal RNG Pool Management
 * ============================================================================
 */

curblasStatus_t curblasEnsureRngCapacity(curblasHandle_t handle, size_t requiredStates) {
    if (handle == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    
    if (ctx->deviceId == -1) {
        return CURBLAS_STATUS_NOT_INITIALIZED;
    }
    
    if (ctx->rngStatesCapacity >= requiredStates) {
        return CURBLAS_STATUS_SUCCESS;
    }
    
    if (ctx->rngStates) {
        cudaFree(ctx->rngStates);
        ctx->rngStates = nullptr;
        ctx->rngStatesCapacity = 0;
    }
    
    cudaError_t cudaStatus = cudaMalloc(&ctx->rngStates, 
        requiredStates * sizeof(curandStatePhilox4_32_10_t));
    if (cudaStatus != cudaSuccess) {
        return CURBLAS_STATUS_ALLOC_FAILED;
    }
    
    int blockSize = 256;
    int numBlocks = (requiredStates + blockSize - 1) / blockSize;
    initRngStates<<<numBlocks, blockSize, 0, ctx->stream>>>(
        ctx->rngStates, requiredStates, ctx->seed);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cudaFree(ctx->rngStates);
        ctx->rngStates = nullptr;
        return CURBLAS_STATUS_EXECUTION_FAILED;
    }
    
    ctx->rngStatesCapacity = requiredStates;
    return CURBLAS_STATUS_SUCCESS;
}

curandStatePhilox4_32_10_t* curblasGetRngStates(curblasHandle_t handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    return ctx->rngStates;
}

cublasHandle_t curblasGetCublasHandle(curblasHandle_t handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    return ctx->cublasHandle;
}

curblasStatus_t curblasEnsureWorkspace(curblasHandle_t handle, size_t requiredSize) {
    if (handle == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    
    if (ctx->deviceId == -1) {
        return CURBLAS_STATUS_NOT_INITIALIZED;
    }
    
    if (ctx->workspaceSize >= requiredSize) {
        return CURBLAS_STATUS_SUCCESS;
    }
    
    if (ctx->workspace) {
        cudaFree(ctx->workspace);
        ctx->workspace = nullptr;
        ctx->workspaceSize = 0;
    }
    
    cudaError_t cudaStatus = cudaMalloc(&ctx->workspace, requiredSize);
    if (cudaStatus != cudaSuccess) {
        return CURBLAS_STATUS_ALLOC_FAILED;
    }
    
    ctx->workspaceSize = requiredSize;
    return CURBLAS_STATUS_SUCCESS;
}

void* curblasGetWorkspace(curblasHandle_t handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    return ctx->workspace;
}

/*
 * ============================================================================
 * curblas Configuration Implementation
 * ============================================================================
 */

curblasStatus_t curblasSetAccuracy(curblasHandle_t handle, curblasAccuracy_t accuracy) {
    if (handle == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    if (accuracy < CURBLAS_ACCURACY_HIGH || accuracy > CURBLAS_ACCURACY_CUSTOM) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    ctx->accuracy = accuracy;
    
    return CURBLAS_STATUS_SUCCESS;
}

curblasStatus_t curblasGetAccuracy(curblasHandle_t handle, curblasAccuracy_t* accuracy) {
    if (handle == nullptr || accuracy == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    *accuracy = ctx->accuracy;
    
    return CURBLAS_STATUS_SUCCESS;
}

curblasStatus_t curblasSetSketchType(curblasHandle_t handle, curblasSketchType_t sketchType) {
    if (handle == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    if (sketchType < CURBLAS_SKETCH_GAUSSIAN || sketchType > CURBLAS_SKETCH_AUTO) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    ctx->defaultSketchType = sketchType;
    
    return CURBLAS_STATUS_SUCCESS;
}

curblasStatus_t curblasSetRandomSeed(curblasHandle_t handle, unsigned long long seed) {
    if (handle == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    ctx->seed = seed;
    
    if (ctx->rngStates && ctx->rngStatesCapacity > 0) {
        int blockSize = 256;
        int numBlocks = (ctx->rngStatesCapacity + blockSize - 1) / blockSize;
        initRngStates<<<numBlocks, blockSize, 0, ctx->stream>>>(
            ctx->rngStates, ctx->rngStatesCapacity, ctx->seed);
    }
    
    return CURBLAS_STATUS_SUCCESS;
}

curblasStatus_t curblasSetMathMode(curblasHandle_t handle, curblasMath_t mode) {
    if (handle == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    ctx->mathMode = mode;
    
    return CURBLAS_STATUS_SUCCESS;
}

/*
 * ============================================================================
 * curblas Utility Functions Implementation
 * ============================================================================
 */

const char* curblasGetStatusString(curblasStatus_t status) {
    switch (status) {
        case CURBLAS_STATUS_SUCCESS:          return "CURBLAS_STATUS_SUCCESS";
        case CURBLAS_STATUS_NOT_INITIALIZED:  return "CURBLAS_STATUS_NOT_INITIALIZED";
        case CURBLAS_STATUS_ALLOC_FAILED:     return "CURBLAS_STATUS_ALLOC_FAILED";
        case CURBLAS_STATUS_INVALID_VALUE:    return "CURBLAS_STATUS_INVALID_VALUE";
        case CURBLAS_STATUS_ARCH_MISMATCH:    return "CURBLAS_STATUS_ARCH_MISMATCH";
        case CURBLAS_STATUS_MAPPING_ERROR:    return "CURBLAS_STATUS_MAPPING_ERROR";
        case CURBLAS_STATUS_EXECUTION_FAILED: return "CURBLAS_STATUS_EXECUTION_FAILED";
        case CURBLAS_STATUS_INTERNAL_ERROR:   return "CURBLAS_STATUS_INTERNAL_ERROR";
        case CURBLAS_STATUS_NOT_SUPPORTED:    return "CURBLAS_STATUS_NOT_SUPPORTED";
        case CURBLAS_STATUS_LICENSE_ERROR:    return "CURBLAS_STATUS_LICENSE_ERROR";
        case CURBLAS_STATUS_INSUFFICIENT_WORKSPACE: return "CURBLAS_STATUS_INSUFFICIENT_WORKSPACE";
        default:                              return "Unknown curblas status";
    }
} 