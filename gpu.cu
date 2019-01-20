//
// Created by Emily & Marc on 10/12/2018.
//

#ifdef CUDA_AVAILABLE

#include "gpu.h"

using namespace utils;

namespace gpu {

    template<unsigned int blockSize, typename kernelHandle>
    float gm_call_kernel(kernelHandle &kernel, const float *U, const float *V, float a, int k, unsigned int n) {
        int numBlocks = (n + blockSize - 1) / blockSize;
        float *sumGpu, *divGpu;
        cudaMallocManaged(&sumGpu, n * sizeof(float));
        cudaMallocManaged(&divGpu, n * sizeof(float));

        kernel << < numBlocks, blockSize, 2 * blockSize * sizeof(float) >> > (U, V, a, k, n, sumGpu, divGpu);
        cudaDeviceSynchronize();

        float sumFinal = 0, divFinal = 0;
        for (unsigned int i = 0; i < numBlocks; i++) {
            sumFinal += sumGpu[i];
            divFinal += divGpu[i];
        }

        cudaFree(sumGpu);
        cudaFree(divGpu);

        return sumFinal / divFinal;
    }

    __device__
    inline float ipowf_helper(float base, int exp) {
        float result = 1;
        while (true) {
            if (exp & 1) result *= base;
            exp >>= 1;
            if (!exp) break;
            base *= base;
        }
        return result;
    }

    __device__
    inline float ipowf(float base, int exp) {
        if (exp > 0) return ipowf_helper(base, exp);
        else if (exp == 0) return 1.f;
        else return 1.f / ipowf_helper(base, -exp);
    }

    __global__
    void gm_kernel(const float *U, const float *V, float a, int k, unsigned int n, float *outSum, float *outDiv) {
        extern __shared__ float sdata[];
        float *sdataSum = sdata;
        float *sdataDiv = sdata + blockDim.x;
        unsigned int tid = threadIdx.x;
        unsigned int index = blockIdx.x * blockDim.x + tid;
        unsigned int stride = blockDim.x * gridDim.x;

        sdataSum[tid] = 0;
        sdataDiv[tid] = 0;

        for (unsigned int i = index; i < n; i += stride) {
            sdataSum[tid] += ipowf(V[i] * U[i] - a, k);
            sdataDiv[tid] += V[i];
        }
        __syncthreads();
        if (tid == 0) {
            outSum[blockIdx.x]=0;
            outDiv[blockIdx.x]=0;
            for (unsigned int i = 0; i < blockDim.x; i++) {
                outSum[blockIdx.x] += sdataSum[i];
                outDiv[blockIdx.x] += sdataDiv[i];
            }
        }
    }

    float gm(const float *U, const float *V, float a, int k, unsigned int n) {
        static const unsigned int blockSize = 256;
        return gm_call_kernel<blockSize>(gm_kernel, U, V, a, k, n);
    }

    template<unsigned int blockSize>
    __device__
    void warpReduce(volatile float *sdata, unsigned int tid) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }

    template<unsigned int blockSize>
    __global__
    void
    gm_kernel_optimized(const float *U, const float *V, float a, int k, unsigned int n, float *outSum, float *outDiv) {
        extern __shared__ float sdata[];
        float *sdataSum = sdata;
        float *sdataDiv = sdata + blockDim.x;
        unsigned int tid = threadIdx.x;
        unsigned int index = blockIdx.x * blockDim.x + tid;
        unsigned int stride = blockDim.x * gridDim.x;

        sdataSum[tid] = 0;
        sdataDiv[tid] = 0;

        for (unsigned int i = index; i < n; i += stride) {
            sdataSum[tid] += ipowf(V[i] * U[i] - a, k);
            sdataDiv[tid] += V[i];
        }
        __syncthreads();

        if (blockSize >= 512) {
            if (tid < 256) {
                sdataSum[tid] += sdataSum[tid + 256];
                sdataDiv[tid] += sdataDiv[tid + 256];
            }
            __syncthreads();
        }
        if (blockSize >= 256) {
            if (tid < 128) {
                sdataSum[tid] += sdataSum[tid + 128];
                sdataDiv[tid] += sdataDiv[tid + 128];
            }
            __syncthreads();
        }
        if (blockSize >= 128) {
            if (tid < 64) {
                sdataSum[tid] += sdataSum[tid + 64];
                sdataDiv[tid] += sdataDiv[tid + 64];
            }
            __syncthreads();
        }

        if (tid < 32) {
            warpReduce<blockSize>(sdataSum, tid);
            warpReduce<blockSize>(sdataDiv, tid);
        }
        if (tid == 0) {
            outSum[blockIdx.x] = sdataSum[0];
            outDiv[blockIdx.x] = sdataDiv[0];
        }
    }


    float gm_optimized(const float *U, const float *V, float a, int k, unsigned int n) {
        static const unsigned int blockSize = 256;
        return gm_call_kernel< blockSize >(gm_kernel_optimized<blockSize>, U, V, a, k, n);
    }

    void initParams(benchmarkParams &params) {
        cudaMalloc(&params.U_gpu, params.n * sizeof(float));
        cudaMalloc(&params.W_gpu, params.n * sizeof(float));
        cudaMemcpy(params.U_gpu, params.U, params.n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(params.W_gpu, params.W, params.n * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    void freeParams(benchmarkParams &params) {
        cudaDeviceSynchronize();
        cudaFree(&params.U_gpu);
        cudaFree(&params.W_gpu);
    }
}
#endif