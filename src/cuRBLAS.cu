#include "cuRBLAS/cuRBLAS.cuh"

#include <cuda_runtime.h>

namespace cuRBLAS {

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
            sharedData[tid] += sharedData[tid + s];
            __syncthreads();
        }

        if (tid == 0) {
            output[threadIdx.x] = sharedData[0];
        }
    }

}