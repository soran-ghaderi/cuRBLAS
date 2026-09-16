#pragma once

#include <cuda_runtime.h>

namespace cuRBLAS {

    __global__ void reduceSum(const float* input, float* output, int n);

} // namespace cuRBLAS
