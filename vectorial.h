//
// Created by Emily & Marc on 10/12/2018.
//
#pragma once

#include "utils.h"

#define RUN_BENCHMARK_VECTORIAL(FUNC) \
    FUNC("AVX", UTILS_WRAP(utils::wrapper_seq, vectorial::gm)); \
    FUNC("AVX_par", UTILS_WRAP(utils::wrapper_par, vectorial::gm_parrallel));

namespace vectorial {

    template <typename functionHandle>
    float wrapper(const functionHandle &f, const utils::benchmarkParams &params ) {
        return f(params.U, params.W, params.a, params.k, params.n);
    }

    float gm(const float *U, const float *W, float a, int k, unsigned int n);
    float gm_parrallel(const float *U, const float *W, float a, int k, unsigned int n, unsigned int nthread);
}