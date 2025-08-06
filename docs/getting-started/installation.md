# Installation

## Quick Installation

Currently cuRBLAS is in development. To install and use cuRBLAS:

### Prerequisites

- CUDA Toolkit 12.0 or later
- CMake 3.18 or later
- C++17 compatible compiler

### Building from Source

```bash
git clone https://github.com/cuRBLAS/cuRBLAS.git
cd curblas
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Installation

```bash
sudo make install
```

More detailed installation instructions coming soon. 