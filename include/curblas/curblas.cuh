#pragma once
#include "cuda_runtime.h"
#include <curand_kernel.h>

namespace curblas {

    __global__ void reduceSum(const float *input, float *output, int n);

    __global__ void gaussianSketchOperator(
        curandStatePhilox4_32_10_t* __restrict__ states,
        float* __restrict__ sketch, 
        int rows, int cols, float scale);

} // namespace curblas