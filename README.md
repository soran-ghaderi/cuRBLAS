# cuRBLAS: CUDA Randomized Basic Linear Algebra Subprograms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/soran-ghaderi/cuRBLAS/ci.yml?branch=main)](https://github.com/soran-ghaderi/cuRBLAS/actions/workflows/ci.yml)
[![PyPI Release](https://img.shields.io/pypi/v/cuRBLAS.svg)](https://pypi.org/project/cuRBLAS)
[![codecov](https://codecov.io/gh/soran-ghaderi/cuRBLAS/branch/main/graph/badge.svg)](https://codecov.io/gh/soran-ghaderi/cuRBLAS)

## Overview

**cuRBLAS** is a high-performance GPU-accelerated library for randomized linear algebra operations, designed as a complement to NVIDIA's cuBLAS. It leverages state-of-the-art randomized algorithms to provide significant computational speedups for large-scale linear algebra problems by trading deterministic accuracy for performance through probabilistic methods.

### Why Randomized Linear Algebra?

Traditional linear algebra operations can be computationally expensive for large matrices. Randomized algorithms offer:

- **Performance**: We will update this section when the results are ready
- **Memory Efficiency**: Reduced memory footprint through sketching techniques
- **Scalability**: Better scaling properties for massive datasets
- **Controllable Accuracy**: User-defined precision guarantees (typically 95-99.9% accuracy)

### Key Applications

- **Machine Learning**: Fast approximate matrix operations for neural networks, dimensionality reduction, kernel methods, graph algorithms
- **Scientific Computing**: Large-scale simulations, PDE/ODE/SDE solving, quantum chemistry calculations
- **Computer Graphics**: Real-time rendering, physics simulations

## Features

### Currently Implemented (v0.1.0)

✅ **Core Infrastructure**
- Context management (`curblasCreate`, `curblasDestroy`)
- CUDA stream support (`curblasSetStream`, `curblasGetStream`)
- Configuration management (accuracy levels, sketch types, random seeding)
- Comprehensive error handling and status reporting
- Custom random number generation optimized for CUDA devices
- Version information system

✅ **Configuration API**
- Accuracy levels: High (99.9%), Medium (99%), Low (95%), Custom
- Sketching method selection: Gaussian, Rademacher, SRHT, CountSketch, Sparse, Auto
- Math mode configuration (including Tensor Core support)
- Random seed control for reproducible results

### Planned Features (Header Declarations Only)

🚧 **Level 3 Operations** (API designed, implementation pending)
- Randomized General Matrix Multiply (RGEMM)
- Randomized Singular Value Decomposition (RSVD)  
- Randomized QR Decomposition (RQR)

🚧 **Utility Functions** (API designed, implementation pending)
- Workspace size calculation
- Optimal sketch size recommendations
- Logging callback system

### Sketching Methods (Types Defined)

The library defines support for multiple randomized sketching techniques:

- **Gaussian**: Standard Gaussian random projections
- **Rademacher**: Binary random projections (+1/-1)
- **SRHT**: Subsampled Randomized Hadamard Transform
- **CountSketch**: Hash-based sketching for sparse data
- **Sparse**: Sparse random projections
- **Auto**: Automatic selection based on matrix properties

*Note: Type definitions and enums are complete, but kernel implementations are in development.*

## Prerequisites

Building cuRBLAS requires:

* **C++17-compliant compiler** (GCC 7+, Clang 5+, MSVC 2017+)
* **CMake** `>= 3.9`
* **CUDA Toolkit** (version 11.0 or later) - for GPU acceleration
* **cuBLAS** (included with CUDA Toolkit) - for basic linear algebra operations
* **Random number generation** - custom implementation for sketching algorithms
* **Catch2** testing framework (for building tests)
* **Python** `>= 3.8` (for Python bindings)
* **Doxygen** (optional, for documentation)

## Installation

### Building from Source

```bash
# Clone the repo
git clone https://github.com/soran-ghaderi/cuRBLAS.git
cd curblas

# Create build directory
mkdir build && cd build

# Configure build
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build the library
cmake --build .

# Install (optional)
cmake --build . --target install
```

### Build Options

Customize the build with CMake variables:

```bash
cmake -DBUILD_TESTING=ON \      # Enable test suite (default: ON)
      -DBUILD_DOCS=ON \         # Enable documentation (default: ON)
      -DBUILD_PYTHON=ON \       # Enable Python bindings (default: ON)
      -DCMAKE_BUILD_TYPE=Release ..
```

### Python Installation

Install as a Python package:

```bash
pip install .
```

Or install directly from PyPI (when available):

```bash
pip install curblas
```

## Usage Example

```c
#include <cuRBLAS/curblas.h>

// Create curblas context
curblasHandle_t handle;
curblasStatus_t status = curblasCreate(&handle);
if (status != CURBLAS_STATUS_SUCCESS) {
    printf("Failed to create curblas handle: %s\n", 
           curblasGetStatusString(status));
    return -1;
}

// Configure accuracy (99% accuracy)
curblasSetAccuracy(handle, CURBLAS_ACCURACY_MEDIUM);

// Set random seed for reproducibility
curblasSetRandomSeed(handle, 42);

// Set default sketching method
curblasSetSketchType(handle, CURBLAS_SKETCH_GAUSSIAN);

// Get version information
int version;
curblasGetVersion(handle, &version);
printf("curblas Version: %d\n", version);

// Note: Matrix operations like curblasRgemm are declared 
// in headers but not yet implemented

// Cleanup
curblasDestroy(handle);
```

## Testing

### C++ Tests

```bash
cd build
ctest --verbose
```

### Python Tests

```bash
pip install .
pytest tests/python/
```

## Documentation

Build the documentation locally:

```bash
cmake --build . --target doxygen
```

Then open `doc/html/index.html` in your browser.

## Current Development Status

**What Works:**
- ✅ Complete context management system
- ✅ CUDA stream integration
- ✅ Configuration and parameter setting
- ✅ Error handling and status reporting
- ✅ Random number generator setup
- ✅ Comprehensive test suite for implemented features

**What's Coming Next:**
- 🚧 Core sketching kernel implementations  
- 🚧 Randomized matrix multiplication (RGEMM)
- 🚧 Basic performance benchmarking
- 🚧 Memory management utilities

## Roadmap

### In Progress
- [x] Core infrastructure and context management
- [x] API design and type definitions  
- [x] Basic CUDA integration and testing framework
- [ ] Core sketching kernel implementations
- [ ] RGEMM implementation
- [ ] Memory management utilities
- [ ] Multiple sketching methods implementation
- [ ] Randomized SVD (RSVD)
- [ ] Randomized QR decomposition
- [ ] Adaptive sketch sizing algorithms
- [ ] Performance benchmarking framework
- [ ] Multi-GPU support
- [ ] Level 1 and Level 2 RBLAS operations
- [ ] Python (& maybe other langs?!) bindings
- [ ] Integration with ML frameworks

### More
- [ ] Community contributions
- [ ] Production optimizations
- [ ] Comprehensive benchmarking studies
- [ ] Academic collaborations

## Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/soran-ghaderi/cuRBLAS.git
cd curblas
pip install -r requirements-dev.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use cuRBLAS in your research, please cite:

```bibtex
@software{curblas2024,
  title={cuRBLAS: CUDA Randomized Basic Linear Algebra Subprograms},
  author={Ghaderi, Soran and contributors},
  year={2024},
  url={https://github.com/soran-ghaderi/cuRBLAS}
}
```

## Acknowledgments

- Built on NVIDIA's CUDA and cuBLAS libraries

## Contact

- **Project Lead**: Soran Ghaderi
- **Issues**: [GitHub Issues](https://github.com/soran-ghaderi/cuRBLAS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/soran-ghaderi/cuRBLAS/discussions)

---

**Status**: 🚧 Active Development - Early Alpha Release

## New Features

- **High-performance sketching algorithms** for randomized linear algebra
- **Memory-efficient implementations** that minimize GPU memory usage
- **Streaming support** for out-of-core computations
- **Comprehensive error handling** with meaningful error messages
- **Custom random number generation** optimized for CUDA devices
- **Python bindings** with numpy integration
- **Extensive testing** with unit tests and benchmarks
- **Cross-platform support** (Linux, Windows, macOS)
