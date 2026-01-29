#include "curblas/curblas.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <algorithm>

namespace curblas {

    __global__ void reduceSum(const float *input, float *output, int n) {
        extern __shared__ float sharedData[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sharedData[tid] += sharedData[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[blockIdx.x] = sharedData[0];
        }

        cooperative_groups::grid_group grid = cooperative_groups::this_grid();
        grid.sync();

        if (blockIdx.x == 0 && tid < 32) {
            int numBlocks = gridDim.x;
            float sum = 0.0f;

            for (int i = tid; i < numBlocks; i += 32) {
                sum += output[i];
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            if (tid == 0) {
                output[0] = sum;
            }
        }
    }

    __global__ void gaussianSketchOperator(
            curandStatePhilox4_32_10_t* __restrict__ states,
            float* __restrict__ sketch, 
            int rows, int cols, float scale) {
        int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        int totalElements = rows * cols;
        int idxStart = tIdx * 4;

        if (idxStart < totalElements) {
            float4 r = curand_normal4(&states[tIdx]);

            r.x *= scale;
            r.y *= scale;
            r.z *= scale;
            r.w *= scale;

            sketch[idxStart] = r.x;

            if (idxStart + 1 < totalElements) {
                sketch[idxStart + 1] = r.y;
            }
            
            if (idxStart + 2 < totalElements) {
                sketch[idxStart + 2] = r.z;
            }

            if (idxStart + 3 < totalElements) {
                sketch[idxStart + 3] = r.w;
            }
        }
    }

}