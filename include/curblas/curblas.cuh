#pragma once
#include "cuda_runtime.h"

namespace curblas {

    __global__ void reduceSum(const float *input, float *output, int n);

    __global__ void generateGaussianSketch(float *sketch, int rows, int cols, long long seed, float scale);


    } // namespace curblas