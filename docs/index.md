# Welcome to cuRBLAS

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __High Performance__

    ---

    Optimized CUDA kernels for maximum throughput on NVIDIA GPUs with support for modern GPU architectures.

    [:octicons-arrow-right-24: Getting Started](getting-started/installation.md)

-   :material-chart-timeline:{ .lg .middle } __Randomized Algorithms__

    ---

    Efficient approximation algorithms for large-scale matrix computations with configurable accuracy levels.

    [:octicons-arrow-right-24: User Guide](guide/overview.md)

-   :material-code-braces:{ .lg .middle } __Easy Integration__

    ---

    C API compatible with existing BLAS workflows, plus Python bindings for rapid prototyping.

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-memory:{ .lg .middle } __Memory Efficient__

    ---

    Reduced memory footprint through sketching techniques, enabling computations on larger matrices.

    [:octicons-arrow-right-24: Examples](examples/basic.md)

</div>

## What is cuRBLAS?

**cuRBLAS** (CUDA Randomized BLAS) is a high-performance library for randomized linear algebra operations on NVIDIA GPUs. It provides implementations of randomized algorithms for common matrix operations, offering significant speedups for large-scale computations.

### Key Features

- **Matrix Multiplication (RGEMM)**: Randomized general matrix multiplication with configurable accuracy
- **Singular Value Decomposition (RSVD)**: Fast approximate SVD using randomized methods  
- **QR Decomposition (RQR)**: Efficient randomized QR factorization
- **Multiple Precision Support**: Single and double precision floating-point operations
- **Flexible Sketching**: Various sketching methods (Gaussian, Rademacher, SRHT, CountSketch)
- **CUDA Integration**: Seamless integration with existing CUDA workflows

### Quick Example

=== "C API"

    ```c
    #include <cuRBLAS/curblas.h>
    
    // Create cuRBLAS handle
    curblasHandle_t handle;
    curblasCreate(&handle);
    
    // Set accuracy level
    curblasSetAccuracy(handle, CURBLAS_ACCURACY_HIGH);
    
    // Perform randomized GEMM: C = α·A·B + β·C
    curblasRgemm(handle, 
                 CURBLAS_OP_N, CURBLAS_OP_N,
                 m, n, k,
                 &alpha, A, lda, 
                 B, ldb,
                 &beta, C, ldc,
                 CURBLAS_SKETCH_GAUSSIAN, 0);
    
    // Cleanup
    curblasDestroy(handle);
    ```

=== "Python"

    ```python
    import curblas
    import numpy as np
    
    // Create arrays on GPU
    A = cp.random.random((1000, 500), dtype=cp.float32)
    B = cp.random.random((500, 800), dtype=cp.float32)
    
    // Randomized matrix multiplication
    C = curblas.rgemm(A, B, accuracy='high', sketch='gaussian')
    ```

### Performance Benefits

Randomized algorithms can provide significant performance improvements, especially for:

- **Large matrices** where exact computation is prohibitively expensive
- **Low-rank or approximately low-rank matrices** where high accuracy is maintained
- **Applications** that can tolerate controlled approximation errors
- **Memory-constrained environments** where sketching reduces memory usage

!!! info "When to Use cuRBLAS"

    cuRBLAS is ideal when you need fast matrix computations and can accept small, controlled approximation errors. It's particularly effective for:
    
    - Machine learning applications (PCA, matrix factorization)
    - Scientific computing (solving large linear systems)
    - Data analysis (dimensionality reduction)
    - Signal processing (fast transforms)

## Getting Started

1. **[Install cuRBLAS](getting-started/installation.md)** - Download and install the library
2. **[Quick Start Guide](getting-started/quick-start.md)** - Basic usage examples
3. **[API Reference](api/index.md)** - Complete function documentation
4. **[Examples](examples/basic.md)** - Real-world usage scenarios

## Community & Support

- **GitHub Repository**: [cuRBLAS/cuRBLAS](https://github.com/cuRBLAS/cuRBLAS)
- **Issue Tracker**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Contributing**: Help improve cuRBLAS

---

<div class="grid" markdown>

<div markdown>
### Latest Release

**Version 0.0.1** - Development

Initial development version with core randomized BLAS operations.

[Download](https://github.com/cuRBLAS/cuRBLAS/releases){ .md-button .md-button--primary }
</div>

<div markdown>
### Documentation

Complete API reference and user guides available.

[Browse Docs](api/index.md){ .md-button }
</div>

</div> 