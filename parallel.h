//
// Created by Emily & Marc on 10/12/2018.
//

#pragma once

#include "utils.h"


#define RUN_BENCHMARK_PARALLEL(FUNC) \
    FUNC("openMP",UTILS_WRAP(utils::wrapper_par, parallel::gm_omp)); \
    FUNC("std_thread",UTILS_WRAP(utils::wrapper_par, parallel::gm_std_thread)); \
    FUNC("std_thread_atom",UTILS_WRAP(utils::wrapper_par, parallel::gm_std_thread_atomic));


namespace parallel {

    float gm_omp(const float *U, const float *V, float a, int k, unsigned int n, unsigned int nthread);
    float gm_std_thread(const float *U, const float *V, float a, int k, unsigned int n, unsigned int nthread);
    float gm_std_thread_atomic(const float *U, const float *V, float a, int k, unsigned int n, unsigned int nthread);
}