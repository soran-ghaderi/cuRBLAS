#ifndef CURBLAS_TYPES_H
#define CURBLAS_TYPES_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct curblasContext;
typedef struct curblasContext* curblasHandle_t;

typedef enum {
    CURBLAS_STATUS_SUCCESS          = 0,
    CURBLAS_STATUS_NOT_INITIALIZED  = 1,
    CURBLAS_STATUS_ALLOC_FAILED     = 2,
    CURBLAS_STATUS_INVALID_VALUE    = 3,
    CURBLAS_STATUS_ARCH_MISMATCH    = 4,
    CURBLAS_STATUS_MAPPING_ERROR    = 5,
    CURBLAS_STATUS_EXECUTION_FAILED = 6,
    CURBLAS_STATUS_INTERNAL_ERROR   = 7,
    CURBLAS_STATUS_NOT_SUPPORTED    = 8,
    CURBLAS_STATUS_LICENSE_ERROR    = 9,
    CURBLAS_STATUS_INSUFFICIENT_WORKSPACE = 10
} curblasStatus_t;

/* operation, fill, diag, side, math, compute and algo codes mirror cublas values */
typedef enum {
    CURBLAS_OP_N = 0,
    CURBLAS_OP_T = 1,
    CURBLAS_OP_C = 2
} curblasOperation_t;

typedef enum {
    CURBLAS_SKETCH_GAUSSIAN     = 0,
    CURBLAS_SKETCH_RADEMACHER   = 1,  // +1/-1 entries
    CURBLAS_SKETCH_SRHT         = 2,  // subsampled randomized hadamard transform
    CURBLAS_SKETCH_COUNTSKETCH  = 3,
    CURBLAS_SKETCH_SPARSE       = 4,
    CURBLAS_SKETCH_AUTO         = 5   // picked from matrix shape and sparsity
} curblasSketchType_t;

typedef enum {
    CURBLAS_ACCURACY_HIGH    = 0,  // 99.9%, bigger sketches
    CURBLAS_ACCURACY_MEDIUM  = 1,  // 99%
    CURBLAS_ACCURACY_LOW     = 2,  // 95%, smaller sketches
    CURBLAS_ACCURACY_CUSTOM  = 3   // caller picks the sketch size
} curblasAccuracy_t;

typedef enum {
    CURBLAS_FILL_MODE_LOWER = 0,
    CURBLAS_FILL_MODE_UPPER = 1
} curblasFillMode_t;

typedef enum {
    CURBLAS_DIAG_NON_UNIT = 0,
    CURBLAS_DIAG_UNIT     = 1
} curblasDiagType_t;

typedef enum {
    CURBLAS_SIDE_LEFT  = 0,
    CURBLAS_SIDE_RIGHT = 1
} curblasSideMode_t;

typedef enum {
    CURBLAS_R_16F = 2,
    CURBLAS_C_16F = 6,
    CURBLAS_R_32F = 0,
    CURBLAS_C_32F = 4,
    CURBLAS_R_64F = 1,
    CURBLAS_C_64F = 5,
    CURBLAS_R_8I  = 7,
    CURBLAS_R_8U  = 8,
    CURBLAS_R_32I = 9,
    CURBLAS_R_32U = 10,
    CURBLAS_R_4I  = 11,
    CURBLAS_R_4U  = 12,
} curblasDataType_t;

typedef enum {
    CURBLAS_DEFAULT_MATH     = 0,
    CURBLAS_TENSOR_OP_MATH   = 1,
    CURBLAS_PEDANTIC_MATH    = 2,
    CURBLAS_TF32_TENSOR_OP_MATH = 3,
    CURBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
} curblasMath_t;

typedef enum {
    CURBLAS_COMPUTE_16F           = 64,
    CURBLAS_COMPUTE_16F_PEDANTIC  = 65,
    CURBLAS_COMPUTE_32F           = 68,
    CURBLAS_COMPUTE_32F_PEDANTIC  = 69,
    CURBLAS_COMPUTE_32F_FAST_16F  = 74,
    CURBLAS_COMPUTE_32F_FAST_16BF = 75,
    CURBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CURBLAS_COMPUTE_64F           = 70,
    CURBLAS_COMPUTE_64F_PEDANTIC  = 71,
    CURBLAS_COMPUTE_32I           = 72,
    CURBLAS_COMPUTE_32I_PEDANTIC  = 73
} curblasComputeType_t;

typedef enum {
    CURBLAS_GEMM_ALGO_0  = 0,
    CURBLAS_GEMM_ALGO_1  = 1,
    CURBLAS_GEMM_ALGO_2  = 2,
    CURBLAS_GEMM_ALGO_3  = 3,
    CURBLAS_GEMM_ALGO_4  = 4,
    CURBLAS_GEMM_ALGO_5  = 5,
    CURBLAS_GEMM_ALGO_6  = 6,
    CURBLAS_GEMM_ALGO_7  = 7,
    CURBLAS_GEMM_DEFAULT = -1,
    CURBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CURBLAS_GEMM_ALGO0_TENSOR_OP   = 100,
    CURBLAS_GEMM_ALGO1_TENSOR_OP   = 101,
    CURBLAS_GEMM_ALGO2_TENSOR_OP   = 102,
    CURBLAS_GEMM_ALGO3_TENSOR_OP   = 103,
    CURBLAS_GEMM_ALGO4_TENSOR_OP   = 104
} curblasGemmAlgo_t;

typedef void (*curblasRandomGenerator_t)(void* state, void* output, size_t count);
typedef void (*curblasLogCallback_t)(int logLevel, const char* functionName, const char* message);

#define CURBLAS_AUTO_SKETCH_SIZE 0

#define CURBLAS_VERSION_MAJOR 0
#define CURBLAS_VERSION_MINOR 1
#define CURBLAS_VERSION_PATCH 1

#ifdef __cplusplus
}
#endif

#endif // CURBLAS_TYPES_H
