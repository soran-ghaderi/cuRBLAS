# cuRBLAS: CUDA Randomized Basic Linear Algebra Subprograms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status: early alpha.** The public header declares the full target API, but most of it is
> not implemented yet. `progress/status.md` is the authoritative ledger of what actually
> works; this README documents only implemented functionality.

## Overview

**cuRBLAS** is a GPU-native library for randomized linear algebra: RandBLAS sketching
semantics expressed in the cuBLAS idiom, with RandLAPACK-style drivers layered on top.
Randomized methods trade a controlled amount of accuracy for large reductions in work and
memory traffic on problems where exact factorizations are too expensive.

RandBLAS is CPU/OpenMP-only, cuSOLVER ships a single randomized driver (`gesvdr`) and no
sketching primitives, so today every GPU randomized-NLA paper hand-rolls its kernels.
cuRBLAS aims to be the missing shared layer.

### Design commitments

- **cuBLAS idiom**: `curblas` prefix, opaque handle first argument, `curblasStatus_t` returns,
  stream and workspace owned by the handle, `extern "C"` boundary.
- **RandBLAS semantics**: distributions, operator descriptors, GEMM-like sketch application,
  submatrix support, so cuRBLAS reads as "RandBLAS, on device".
- **Reproducibility**: same seed and dimensions produce a bitwise-identical sketch regardless
  of launch geometry or GPU model.
- **No performance claims without benchmarks**: numbers land in this README only when a
  harness measures them against naive composition (curand + cuBLAS).

## What works today

| Area | Status |
|---|---|
| Context and handle lifecycle (`curblasCreate`, `curblasDestroy`, `curblasGetVersion`) | ✅ implemented and tested |
| CUDA stream get/set (`curblasSetStream`, `curblasGetStream`) | ✅ implemented and tested |
| Configuration: accuracy, sketch type, random seed, math mode | ✅ stored on the handle, not yet consumed by kernels |
| Status strings (`curblasGetStatusString`) | ✅ implemented and tested |
| Workspace and RNG-state management (`curblasEnsureWorkspace`, `curblasEnsureRngCapacity`) | ✅ implemented and tested |
| `curblas::gaussianSketchOperator` device kernel (float4-vectorized) | ✅ implemented, statistically tested |
| `curblas::reduceSum` device kernel | ✅ implemented and tested (cooperative-groups demo) |

### Declared in the headers but not implemented

`curblasRgemm` / `curblasDrgemm`, `curblasRsvd` / `curblasDrsvd`, `curblasRqr`,
`curblasGetWorkspaceSize`, `curblasSetLoggerCallback`, `curblasGetOptimalSketchSize`.

Of the sketch types in `curblasSketchType_t`, only `CURBLAS_SKETCH_GAUSSIAN` has a kernel
behind it; `RADEMACHER`, `SRHT`, `COUNTSKETCH`, `SPARSE` and `AUTO` are enum values only.
Calling an unimplemented entry point does not perform the operation.

## Prerequisites

* C++17-compliant compiler
* CMake `>= 3.18`
* CUDA Toolkit 11.0 or later, including cuBLAS and cuRAND (optional at configure time: without
  CUDA a reduced CPU-only target builds, but no kernel runs)
* [Catch2](https://github.com/catchorg/Catch2) v3, for the test suite
* Doxygen, optional, for API documentation

An NVIDIA GPU is required to *run* the kernel tests; the build itself does not need one.

Target GPU architectures are selected automatically from the toolkit version, since each
toolkit supports a different range (Pascal and Volta are removed in CUDA 13; Blackwell needs
CUDA 12.8 or later). Configure prints the resulting list. Override it with
`-DCMAKE_CUDA_ARCHITECTURES=<list>`, for example `-DCMAKE_CUDA_ARCHITECTURES=120` to build
only for a Blackwell consumer card.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Useful options:

```bash
-DBUILD_TESTING=OFF   # skip the test suite (default: ON; needs Catch2 v3)
-DBUILD_DOCS=OFF      # skip Doxygen documentation (default: ON)
-DBUILD_PYTHON=ON     # Python bindings (default: OFF; see the note below)
```

## Usage

### Generating a Gaussian sketch operator

The high-level sketch API is not written yet, so the kernel is called directly. RNG states
live on the handle, so ask for enough of them before launching. A complete, compiling version
of this program is in [`app/gaussian_sketch.cu`](app/gaussian_sketch.cu).

```cpp
#include "curblas/curblas.h"
#include "curblas/curblas.cuh"
#include <cuda_runtime.h>

int rows = 4000, cols = 5000;
int totalElements = rows * cols;
float scale = 1.0f;

curblasHandle_t handle = nullptr;
curblasCreate(&handle);

// each thread emits 4 floats, so size the RNG state pool to the thread count
int blockSize = 256;
int totalThreads = (totalElements + 3) / 4;
int numBlocks = (totalThreads + blockSize - 1) / blockSize;
curblasEnsureRngCapacity(handle, totalThreads);

float* d_sketch = nullptr;
cudaMalloc(&d_sketch, totalElements * sizeof(float));

curandStatePhilox4_32_10_t* states = curblasGetRngStates(handle);
curblas::gaussianSketchOperator<<<numBlocks, blockSize>>>(states, d_sketch, rows, cols, scale);

cudaFree(d_sketch);
curblasDestroy(handle);
```

> The RNG state pool is scheduled for removal: sketch elements will be derived statelessly
> from a counter-based Philox stream, which is what makes the output independent of launch
> geometry. This kernel signature will change when that lands.

### Handle and configuration

```c
#include "curblas/curblas.h"

curblasHandle_t handle;
curblasStatus_t status = curblasCreate(&handle);
if (status != CURBLAS_STATUS_SUCCESS) {
    printf("failed to create curblas handle: %s\n", curblasGetStatusString(status));
    return -1;
}

curblasSetAccuracy(handle, CURBLAS_ACCURACY_MEDIUM);
curblasSetSketchType(handle, CURBLAS_SKETCH_GAUSSIAN);
curblasSetRandomSeed(handle, 42);

int version;
curblasGetVersion(handle, &version);

curblasDestroy(handle);
```

These settings are recorded on the handle and read back correctly; they will drive kernel
selection once the sketch API exists.

## Testing

```bash
ctest --test-dir build          # or run ./build/tests/tests directly
```

Kernel tests are skipped when no CUDA device is present. Randomized kernels are validated
statistically (fixed seed, distribution moments within tolerance), never against golden
values.

Continuous integration is intentionally disabled: the test suite needs a GPU and no GPU
runner is configured. Run `ctest` locally before pushing.

## Documentation

```bash
cmake --build build --target doxygen
```

Output lands in `docs/api/doxygen/html`.

## Python bindings

The `python/` directory is a non-functional template leftover: the module is empty and its
`__init__.py` imports a symbol that does not exist. Bindings are parked until the core
sketching layer is stable, and `BUILD_PYTHON` is off by default. There is no PyPI release.

## Roadmap

Ordered by phase; `progress/plans.md` holds the detail and done-conditions, `progress/status.md`
the live checkboxes.

- **Phase 1: dense Tier 0.** Stateless counter-based Philox, Gaussian + Rademacher + uniform
  operators, distribution/operator descriptors, GEMM-like `curblas{S,D}sketch` built on
  workspace generation plus cuBLAS, reproducibility tests.
- **Phase 2: sparse and SRHT.** CountSketch/SJLT, batched fast Walsh-Hadamard transform,
  parity of left/right/submatrix application across every operator type.
- **Phase 3: fused path and benchmarks.** Benchmark harness first, then generate-and-multiply
  fused kernels with the workspace path as the correctness oracle, plus TF32/FP16 paths.
- **Phase 4: drivers.** RGEMM, RSVD (measured against `cusolverDnXgesvdr`), RQR, sketched
  least squares.
- **Phase 5: ecosystem.** Real Python bindings with DLPack interop, batched APIs, CUDA graph
  capture audit, multi-GPU exploration.

## Contributing

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE.md](LICENSE.md).

## Citation

```bibtex
@software{curblas2024,
  title={cuRBLAS: CUDA Randomized Basic Linear Algebra Subprograms},
  author={Ghaderi, Soran and contributors},
  year={2024},
  url={https://github.com/soran-ghaderi/cuRBLAS}
}
```

## Contact

- **Project lead**: Soran Ghaderi
- **Issues**: [GitHub Issues](https://github.com/soran-ghaderi/cuRBLAS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/soran-ghaderi/cuRBLAS/discussions)
