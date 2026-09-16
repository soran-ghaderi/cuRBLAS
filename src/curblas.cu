#include "curblas/curblas.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace curblas {

    __global__ __launch_bounds__(256, 4)
    void reduceSum(const float *input, float *output, int n) {
        extern __shared__ float sharedData[];

        int tid = threadIdx.x;
        int gridStride = blockDim.x * gridDim.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        float localSum = 0.0f;
        for (int i = idx; i < n; i += gridStride) {
            localSum += input[i];
        }
        sharedData[tid] = localSum;

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

    __global__ __launch_bounds__(256, 4)
    void gaussianSketchOperator(
            curandStatePhilox4_32_10_t* __restrict__ states,
            float* __restrict__ sketch, 
            int rows, int cols, float scale) {
        int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        int gridStride = blockDim.x * gridDim.x;
        int totalElements = rows * cols;
        int numFloat4 = totalElements >> 2;
        int remainder = totalElements & 3;
        float4* __restrict__ sketch4 = reinterpret_cast<float4*>(sketch);

        curandStatePhilox4_32_10_t localState = states[tIdx];

        for (int i = tIdx; i < numFloat4; i += gridStride) {
            float4 r = curand_normal4(&localState);
            r.x *= scale;
            r.y *= scale;
            r.z *= scale;
            r.w *= scale;
            sketch4[i] = r;
        }

        if (tIdx == 0 && remainder > 0) {
            int baseIdx = numFloat4 << 2;
            float4 r = curand_normal4(&localState);
            sketch[baseIdx] = r.x * scale;
            if (remainder > 1) sketch[baseIdx + 1] = r.y * scale;
            if (remainder > 2) sketch[baseIdx + 2] = r.z * scale;
        }

        states[tIdx] = localState;
    }

}