//
// Created by Emily & Marc on 10/12/2018.
//
#include <cmath>

#include "sequential.h"
#include "utils.h"
using namespace utils;

#include <iostream>

namespace sequential {
    float gm(const float *U, const float *V, float a, int k, unsigned int n) {
        float accumulateSum = 0, accumulateDiv = 0;
        for (unsigned int i = 0; i < n; i++) {
            accumulateSum += std::pow(V[i] * U[i] - a, static_cast<float>(k));
            accumulateDiv += V[i];
        }
        return accumulateSum / accumulateDiv;
    }

    float gm_optimized_power(const float *U, const float *V, float a, int k, unsigned int n) {
        float accumulateSum = 0, accumulateDiv = 0;
        for (unsigned int i = 0; i < n; i++) {
            accumulateSum += ipowf(V[i] * U[i] - a, k);
            accumulateDiv += V[i];
        }
        return accumulateSum / accumulateDiv;
    }
}
