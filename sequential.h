//
// Created by Emily & Marc on 10/12/2018.
//

#pragma once

#include "utils.h"


#define RUN_BENCHMARK_SEQUENTIAL(FUNC) \
    FUNC("ipow", UTILS_WRAP(wrapper_seq, sequential::gm_optimized_power));

namespace sequential {

    float gm(const float *U, const float *V, float a, int k, unsigned int n);

    float gm_optimized_power(const float *U, const float *V, float a, int k, unsigned int n);
}
