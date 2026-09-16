# Welcome to cuRBLAS

!!! warning "Early alpha"

    cuRBLAS is under active development. The public header declares the full target API, but
    most of it is not implemented yet. This documentation covers only what is implemented;
    anything absent here is not available.

**cuRBLAS** (CUDA Randomized BLAS) is a GPU-native library for randomized linear algebra:
RandBLAS sketching semantics expressed in the cuBLAS idiom, with RandLAPACK-style drivers
planned on top.

<div class="grid cards" markdown>

-   :material-memory:{ .lg .middle } __GPU-native sketching__

    ---

    Randomized sketching primitives written as CUDA kernels, with the handle owning the
    stream, cuBLAS handle and workspace.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-dice-multiple:{ .lg .middle } __Reproducible randomness__

    ---

    Counter-based Philox generation: the same seed and dimensions yield the same sketch,
    independent of launch geometry.

    [:octicons-arrow-right-24: API Reference](api/index.md)

</div>

## What is implemented

| Area | Status |
|---|---|
| Context and handle lifecycle, version query | ✅ implemented and tested |
| CUDA stream get/set | ✅ implemented and tested |
| Configuration: accuracy, sketch type, random seed, math mode | ✅ stored on the handle, not yet consumed by kernels |
| Status strings | ✅ implemented and tested |
| Workspace and RNG-state management | ✅ implemented and tested |
| `curblas::gaussianSketchOperator` device kernel | ✅ implemented, statistically tested |
| `curblas::reduceSum` device kernel | ✅ implemented and tested |

Declared in the headers but **not implemented**: `curblasRgemm`/`curblasDrgemm`,
`curblasRsvd`/`curblasDrsvd`, `curblasRqr`, `curblasGetWorkspaceSize`,
`curblasSetLoggerCallback`, `curblasGetOptimalSketchSize`. Of the sketch types in
`curblasSketchType_t`, only `CURBLAS_SKETCH_GAUSSIAN` has a kernel behind it.

There are no Python bindings; the `python/` directory is a non-functional template leftover.

## Quick example

Creating a handle and configuring it works today:

```c
#include "curblas/curblas.h"

curblasHandle_t handle;
if (curblasCreate(&handle) != CURBLAS_STATUS_SUCCESS) {
    return -1;
}

curblasSetAccuracy(handle, CURBLAS_ACCURACY_MEDIUM);
curblasSetSketchType(handle, CURBLAS_SKETCH_GAUSSIAN);
curblasSetRandomSeed(handle, 42);

curblasDestroy(handle);
```

For a complete program that generates a Gaussian sketch operator on the device, see
`app/gaussian_sketch.cu` in the repository.

## Why randomized linear algebra

Randomized algorithms approximate expensive matrix computations by first projecting the
problem into a much smaller space. They are effective when:

- matrices are large enough that exact computation is prohibitive
- the matrix is low-rank or approximately low-rank, so little information is lost
- the application tolerates a small, controlled approximation error
- memory, not arithmetic, is the binding constraint

Performance figures will be published here once a benchmark harness measures them against
naive composition (cuRAND generation followed by a cuBLAS GEMM). Until then cuRBLAS makes no
speedup claims.

## Getting started

1. **[Install cuRBLAS](getting-started/installation.md)**
2. **[API Reference](api/index.md)**

## Community

- **GitHub repository**: [soran-ghaderi/cuRBLAS](https://github.com/soran-ghaderi/cuRBLAS)
- **Issue tracker**: [report bugs and request features](https://github.com/soran-ghaderi/cuRBLAS/issues)
- **Discussions**: [ask questions](https://github.com/soran-ghaderi/cuRBLAS/discussions)
