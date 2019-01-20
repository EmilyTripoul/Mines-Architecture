//
// Created by Emily & Marc on 10/12/2018.
//

#pragma once
#ifdef CUDA_AVAILABLE

#include "utils.h"

#define RUN_BENCHMARK_GPU(FUNC) \
    FUNC("cuda", UTILS_WRAP(gpu::wrapper_cuda, gpu::gm)); \
    FUNC("cuda_op", UTILS_WRAP(gpu::wrapper_cuda, gpu::gm_optimized));

namespace gpu {

    template <typename functionHandle>
    float wrapper_cuda(const functionHandle &f, const utils::benchmarkParams &params ) {
        return f(params.U_gpu, params.W_gpu, params.a, params.k, params.n);
    }


    float gm(const float *U, const float *V, float a, int k, unsigned int n);
    float gm_optimized(const float *U, const float *V, float a, int k, unsigned int n);


    void initParams(utils::benchmarkParams &params);
    void freeParams(utils::benchmarkParams &params);

};
#endif
