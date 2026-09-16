# cuRBLAS

Randomized linear algebra on the GPU: sketching primitives with RandBLAS semantics, written in the cuBLAS style.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Early alpha.** The header declares the whole planned API, but most of it is not written yet.
> See [what works](#what-works) before you depend on anything.

Randomized methods shrink a large matrix problem by multiplying it with a small random matrix
(a *sketch*). You give up a little accuracy and save a lot of time and memory. RandBLAS does
this on the CPU. cuSOLVER ships one randomized routine and no sketching primitives at all, so
most GPU code in this area is hand-rolled. cuRBLAS is meant to fill that gap.

## What works

- handle and stream: `curblasCreate`, `curblasDestroy`, `curblasSetStream`, `curblasGetStream`, `curblasGetVersion`
- settings kept on the handle: accuracy, sketch type, seed, math mode (no kernel reads them yet)
- workspace and RNG state allocation
- two device kernels: `gaussianSketchOperator` and `reduceSum`

All of the above has tests.

## What does not

These are declared in the header and do nothing if you call them:
`curblasRgemm`, `curblasDrgemm`, `curblasRsvd`, `curblasDrsvd`, `curblasRqr`,
`curblasGetWorkspaceSize`, `curblasSetLoggerCallback`, `curblasGetOptimalSketchSize`.

Only `CURBLAS_SKETCH_GAUSSIAN` has a kernel behind it. Rademacher, SRHT, CountSketch, sparse
and auto are enum values and nothing more.

No speed numbers either. There is no benchmark harness yet, so this README makes no
performance claims.

## Build

You need a C++17 compiler, CMake 3.18 or newer, and the CUDA toolkit (cuBLAS and cuRAND come
with it). Catch2 v3 for the tests, Doxygen for the docs.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

Handy flags: `-DBUILD_TESTING=OFF`, `-DBUILD_DOCS=OFF`, `-DBUILD_PYTHON=ON` (off by default).

GPU architectures are chosen from your toolkit version and printed when you configure. Pin
them yourself with something like `-DCMAKE_CUDA_ARCHITECTURES=120`.

Building does not need a GPU, running the tests does. CI is switched off because there is no
GPU runner, so run `ctest` yourself before pushing.

## Example

The sketch API is not written yet, so you launch the kernel directly. Full program in
[app/gaussian_sketch.cu](app/gaussian_sketch.cu).

```cpp
curblasHandle_t handle;
curblasCreate(&handle);

int threads = (rows * cols + 3) / 4;   // one thread writes four floats
curblasEnsureRngCapacity(handle, threads);

curblas::gaussianSketchOperator<<<blocks, 256>>>(
    curblasGetRngStates(handle), d_sketch, rows, cols, 1.0f);

curblasDestroy(handle);
```

That signature is going to change. The RNG state pool is being replaced by stateless Philox,
so a sketch will depend on the seed alone and not on how you launched the kernel.

## Roadmap

1. dense sketching: stateless Philox, Gaussian and Rademacher operators, a GEMM-like `curblasSsketch`
2. sparse and SRHT sketches
3. fused generate-and-multiply kernels, and a benchmark harness to prove they are worth it
4. drivers: RGEMM, RSVD, RQR, sketched least squares
5. Python bindings

## Python

`python/` is a leftover from the project template. It does not work and is off by default.

## License

MIT, see [LICENSE.md](LICENSE.md).

## Citation

```bibtex
@software{ghaderi2024curblas,
  title={cuRBLAS: CUDA Randomized Basic Linear Algebra Subprograms},
  author={Soran Ghaderi},
  year={2024},
  url={https://github.com/soran-ghaderi/cuRBLAS}
}
```

## Contact

Questions and bugs: [issues](https://github.com/soran-ghaderi/cuRBLAS/issues) and
[discussions](https://github.com/soran-ghaderi/cuRBLAS/discussions).
