#include "curblas/curblas.h"
#include "curblas/curblas_types.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <memory>
#include <cmath>

// Define M_PI if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// for the rand generator, I decided to go with the CURAND lib
///**
// * Custom random number generator state for CUDA
// */
//struct curblasRngState {
//    unsigned long long seed;
//    unsigned long long counter;
//
//    __device__ __host__ curblasRngState(unsigned long long s = 1234ULL) : seed(s), counter(0) {}
//
//    // Simple PRNG based on linear congruential generator
//    __device__ __host__ unsigned int next() {
//        counter = (counter * 1103515245ULL + 12345ULL) ^ seed;
//        return (unsigned int)(counter >> 16);
//    }
//
//    // Generate float in [0, 1)
//    __device__ __host__ float nextFloat() {
//        return (float)next() / (float)UINT_MAX;
//    }
//
//    // Generate normal distributed float (Box-Muller transform)
//    __device__ __host__ float nextGaussian() {
//        static bool hasSpare = false;
//        static float spare;
//
//        if (hasSpare) {
//            hasSpare = false;
//            return spare;
//        }
//
//        hasSpare = true;
//        float u = nextFloat();
//        float v = nextFloat();
//
//        // Use device-compatible math functions
//        #ifdef __CUDA_ARCH__
//        float mag = sqrtf(-2.0f * logf(u));
//        spare = mag * cosf(2.0f * (float)M_PI * v);
//        return mag * sinf(2.0f * (float)M_PI * v);
//        #else
//        float mag = sqrt(-2.0f * log(u));
//        spare = mag * cos(2.0f * (float)M_PI * v);
//        return mag * sin(2.0f * (float)M_PI * v);
//        #endif
//    }
//};

__global__ void setup_rng_kernel(curandState *state, unsigned long long seed) {
    curand_init(seed, threadIdx.x, 0, state);
}


/**
 * Internal curblas context structure
 */
struct curblasContext {
    // CUDA stream for operations
    cudaStream_t stream;
    bool ownsStream;
    
    // Custom random number generation
//    curblasRngState* rng;
    curandState* rng;
    unsigned long long seed;
    
    // Configuration
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
        rng(nullptr),
        seed(1234ULL),
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
            ctx->rng = nullptr;
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
        
        // Initialize custom random number generator
//        ctx->rng = new curblasRngState(ctx->seed);
//        if (!ctx->rng) {
//            cudaStreamDestroy(ctx->stream);
//            delete ctx;
//            return CURBLAS_STATUS_ALLOC_FAILED;
//        }
//
        cudaStatus = cudaMalloc(&ctx->rng, sizeof(curandState));
        if (cudaStatus != cudaSuccess) {
            cudaStreamDestroy(ctx->stream);
            delete ctx;
            return CURBLAS_STATUS_ALLOC_FAILED;
        }

        // Initialize the RNG state using our kernel
        setup_rng_kernel<<<1, 1, 0, ctx->stream>>>(ctx->rng, ctx->seed);
        cudaStatus = cudaGetLastError(); // Check for kernel launch errors
        if (cudaStatus != cudaSuccess) {
            cudaFree(ctx->rng);
            cudaStreamDestroy(ctx->stream);
            delete ctx;
            return CURBLAS_STATUS_EXECUTION_FAILED;
        }


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
    
    // Destroy custom random number generator
    if (ctx->rng) {
//        delete ctx->rng;
        cudaFree(ctx->rng);

    }
    
    // Destroy stream if we own it
    if (ctx->stream && ctx->ownsStream) {
        cudaStreamDestroy(ctx->stream);
    }
    
    // Free context
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
    ctx->ownsStream = false;  // We don't own external streams
    
    // Note: No need to update stream for custom RNG since it's not tied to CUDA streams
    
    return CURBLAS_STATUS_SUCCESS;
}

curblasStatus_t curblasGetStream(curblasHandle_t handle, cudaStream_t* streamId) {
    if (handle == nullptr || streamId == nullptr) {
        return CURBLAS_STATUS_INVALID_VALUE;
    }
    
    curblasContext* ctx = static_cast<curblasContext*>(handle);
    
    // For CPU-only contexts, return null stream
    if (ctx->deviceId == -1) {
        *streamId = nullptr;
        return CURBLAS_STATUS_SUCCESS;
    }
    
    *streamId = ctx->stream;
    
    return CURBLAS_STATUS_SUCCESS;
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
    
    // Update custom random number generator seed
    if (ctx->rng) {
//        ctx->rng->seed = seed;
//        ctx->rng->counter = 0;  // Reset counter when seed changes
        setup_rng_kernel<<<1, 1, 0, ctx->stream>>>(ctx->rng, ctx->seed);

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