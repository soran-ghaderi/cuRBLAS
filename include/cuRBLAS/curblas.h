#ifndef CURBLAS_H
#define CURBLAS_H

#include "curblas_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================================
 * cuRBLAS Context Management
 * ============================================================================
 */

/**
 * @brief Create a cuRBLAS context handle
 * @param handle Pointer to cuRBLAS handle to be created
 * @return cuRBLAS status code
 */
curblasStatus_t curblasCreate(curblasHandle_t* handle);

/**
 * @brief Destroy a cuRBLAS context handle
 * @param handle cuRBLAS handle to be destroyed
 * @return cuRBLAS status code
 */
curblasStatus_t curblasDestroy(curblasHandle_t handle);

/**
 * @brief Get version information
 * @param version Pointer to store version number
 * @return cuRBLAS status code
 */
curblasStatus_t curblasGetVersion(curblasHandle_t handle, int* version);

/**
 * @brief Set CUDA stream for cuRBLAS operations
 * @param handle cuRBLAS handle
 * @param streamId CUDA stream
 * @return cuRBLAS status code
 */
curblasStatus_t curblasSetStream(curblasHandle_t handle, cudaStream_t streamId);

/**
 * @brief Get CUDA stream used by cuRBLAS
 * @param handle cuRBLAS handle
 * @param streamId Pointer to CUDA stream
 * @return cuRBLAS status code
 */
curblasStatus_t curblasGetStream(curblasHandle_t handle, cudaStream_t* streamId);

/*
 * ============================================================================
 * cuRBLAS Configuration and Control
 * ============================================================================
 */

/**
 * @brief Set global accuracy level for randomized operations
 * @param handle cuRBLAS handle
 * @param accuracy Accuracy level (affects sketch sizes)
 * @return cuRBLAS status code
 */
curblasStatus_t curblasSetAccuracy(curblasHandle_t handle, curblasAccuracy_t accuracy);

/**
 * @brief Get current accuracy level
 * @param handle cuRBLAS handle
 * @param accuracy Pointer to store current accuracy level
 * @return cuRBLAS status code
 */
curblasStatus_t curblasGetAccuracy(curblasHandle_t handle, curblasAccuracy_t* accuracy);

/**
 * @brief Set default sketching method
 * @param handle cuRBLAS handle
 * @param sketchType Default sketching method
 * @return cuRBLAS status code
 */
curblasStatus_t curblasSetSketchType(curblasHandle_t handle, curblasSketchType_t sketchType);

/**
 * @brief Set random seed for reproducible results
 * @param handle cuRBLAS handle
 * @param seed Random seed value
 * @return cuRBLAS status code
 */
curblasStatus_t curblasSetRandomSeed(curblasHandle_t handle, unsigned long long seed);

/**
 * @brief Set math mode (tensor core usage)
 * @param handle cuRBLAS handle
 * @param mode Math mode
 * @return cuRBLAS status code
 */
curblasStatus_t curblasSetMathMode(curblasHandle_t handle, curblasMath_t mode);

/*
 * ============================================================================
 * cuRBLAS Level 3: Randomized Matrix-Matrix Operations
 * ============================================================================
 */

/**
 * @brief Randomized single precision general matrix multiply
 * 
 * Computes C = alpha * op(A) * op(B) + beta * C using randomized algorithms
 * 
 * @param handle cuRBLAS handle
 * @param transa Operation on matrix A
 * @param transb Operation on matrix B
 * @param m Number of rows of op(A) and C
 * @param n Number of columns of op(B) and C
 * @param k Number of columns of op(A) and rows of op(B)
 * @param alpha Scalar alpha
 * @param A Matrix A on device
 * @param lda Leading dimension of A
 * @param B Matrix B on device
 * @param ldb Leading dimension of B
 * @param beta Scalar beta
 * @param C Matrix C on device
 * @param ldc Leading dimension of C
 * @param sketchType Sketching method to use
 * @param sketchSize Sketch size (0 for automatic)
 * @return cuRBLAS status code
 */
curblasStatus_t curblasRgemm(
    curblasHandle_t handle,
    curblasOperation_t transa, curblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc,
    curblasSketchType_t sketchType,
    int sketchSize
);

/**
 * @brief Randomized double precision general matrix multiply
 */
curblasStatus_t curblasDrgemm(
    curblasHandle_t handle,
    curblasOperation_t transa, curblasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc,
    curblasSketchType_t sketchType,
    int sketchSize
);

/*
 * ============================================================================
 * cuRBLAS Randomized Decompositions
 * ============================================================================
 */

/**
 * @brief Randomized Singular Value Decomposition (single precision)
 * 
 * Computes approximate SVD: A ≈ U * S * V^T using randomized algorithms
 * 
 * @param handle cuRBLAS handle
 * @param m Number of rows of A
 * @param n Number of columns of A
 * @param targetRank Target rank for approximation
 * @param A Input matrix A (m x n)
 * @param lda Leading dimension of A
 * @param U Left singular vectors (m x targetRank)
 * @param ldu Leading dimension of U
 * @param S Singular values (targetRank)
 * @param Vt Right singular vectors transposed (targetRank x n)
 * @param ldvt Leading dimension of Vt
 * @param residualNorm Pointer to store approximation error (optional)
 * @return cuRBLAS status code
 */
curblasStatus_t curblasRsvd(
    curblasHandle_t handle,
    int m, int n, int targetRank,
    const float* A, int lda,
    float* U, int ldu,
    float* S,
    float* Vt, int ldvt,
    float* residualNorm
);

/**
 * @brief Randomized Singular Value Decomposition (double precision)
 */
curblasStatus_t curblasDrsvd(
    curblasHandle_t handle,
    int m, int n, int targetRank,
    const double* A, int lda,
    double* U, int ldu,
    double* S,
    double* Vt, int ldvt,
    double* residualNorm
);

/**
 * @brief Randomized QR decomposition (single precision)
 * 
 * Computes approximate QR factorization: A ≈ Q * R
 * 
 * @param handle cuRBLAS handle
 * @param m Number of rows of A
 * @param n Number of columns of A
 * @param A Input matrix A, overwritten with Q
 * @param lda Leading dimension of A
 * @param tau Scalar factors of elementary reflectors
 * @param workspaceSize Size of workspace required
 * @param workspace Workspace array
 * @return cuRBLAS status code
 */
curblasStatus_t curblasRqr(
    curblasHandle_t handle,
    int m, int n,
    float* A, int lda,
    float* tau,
    size_t workspaceSize,
    float* workspace
);

/*
 * ============================================================================
 * cuRBLAS Utility Functions
 * ============================================================================
 */

/**
 * @brief Get the string description of a cuRBLAS status code
 * @param status cuRBLAS status code
 * @return String description of the status
 */
const char* curblasGetStatusString(curblasStatus_t status);

/**
 * @brief Get workspace size required for an operation
 * @param handle cuRBLAS handle
 * @param operation Operation type
 * @param m Matrix dimension m
 * @param n Matrix dimension n
 * @param k Matrix dimension k (for GEMM)
 * @param dataType Data type
 * @param workspaceSize Pointer to store required workspace size
 * @return cuRBLAS status code
 */
curblasStatus_t curblasGetWorkspaceSize(
    curblasHandle_t handle,
    int operation,
    int m, int n, int k,
    curblasDataType_t dataType,
    size_t* workspaceSize
);

/**
 * @brief Set logging callback function
 * @param callback Callback function for logging
 * @return cuRBLAS status code
 */
curblasStatus_t curblasSetLoggerCallback(curblasLogCallback_t callback);

/**
 * @brief Get optimal sketch size recommendation
 * @param handle cuRBLAS handle
 * @param m Matrix rows
 * @param n Matrix columns
 * @param k Matrix inner dimension
 * @param accuracy Desired accuracy level
 * @param sketchType Sketching method
 * @param optimalSize Pointer to store optimal sketch size
 * @return cuRBLAS status code
 */
curblasStatus_t curblasGetOptimalSketchSize(
    curblasHandle_t handle,
    int m, int n, int k,
    curblasAccuracy_t accuracy,
    curblasSketchType_t sketchType,
    int* optimalSize
);

#ifdef __cplusplus
}
#endif

#endif // CURBLAS_H 