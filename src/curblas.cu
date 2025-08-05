#include "curblas/curblas.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <algorithm>

//struct curblasRngState;

namespace curblas {

//int add_one(int x){
//  return x + 1;
//}

    __global__ void reduceSum(const float *input, float *output, int n) {
        extern __shared__ float sharedData[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid <
                s) { // this solved the problem of output=0. To avoid accessing out of bounds (the shared memory array size)!
                sharedData[tid] += sharedData[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[blockIdx.x] = sharedData[0];
//            atomicAdd(output, sharedData[0]);
        }

        cooperative_groups::grid_group grid = cooperative_groups::this_grid();
        grid.sync();

        // final reduct using first warp of first block

        if (blockIdx.x == 0 && tid < 16) {
            int numBlocks = gridDim.x;
            float sum = 0.0f;

            // Warp-level reduction
            for (int i = tid; i < numBlocks; i += 32) {
                sum += output[i];
            }

            // Reduce within warp using shuffle operations
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            if (tid == 0) {
                output[0] = sum;
            }

        }
    }

    __global__ void generateGaussianSketch(float *sketch, int rows, int cols, long long seed, float scale){
        int tIdx = blockIdx.x * blockDim.x + threadIdx.x;

        int totalElements = rows * cols;
        // if (idx < totalElements) {
        //     curandState state;
        //     curand_init(seed, idx, 0, &state);
        //     sketch[idx] = curand_normal(&state) * scale;
        // }

        // mem coalesced version:

        int idxStart = tIdx * 4;
        // curandState state;
        curandStatePhilox4_32_10_t state;

        curand_init(seed, tIdx, 0, &state);

        if(idxStart < totalElements) {
            float4 r = curand_normal4(&state);

            r.x *= scale;
            r.y *= scale;
            r.z *= scale;
            r.w *= scale;

            sketch[idxStart] = r.x;

            if(idxStart + 1 < totalElements) {
                sketch[idxStart + 1] = r.y;
            }
            
            if(idxStart + 2 < totalElements) {
                sketch[idxStart + 2] = r.z;
            }

            if (idxStart + 3 < totalElements) {
                sketch[idxStart + 3] = r.w;
            }


        }
    }

}