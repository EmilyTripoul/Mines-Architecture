//
// Created by Emily & Marc on 10/12/2018.
//

#include "vectorial.h"
#include "utils.h"

using namespace utils;

#include <iostream>

namespace vectorial {

#ifdef FORCE_ALIGN
#define MM256_LOAD(x) _mm256_load_ps(x)
#else
#define MM256_LOAD(x) _mm256_loadu_ps(x)
#endif

    float gm(const float *U, const float *W, float a, int k, unsigned int n) {
        float accumulateSum = 0, accumulateDiv = 0;
        __m256 aSimd = _mm256_set1_ps(-a);
        unsigned int i;
        for (i = 0; i + AVX_VECTOR_SIZE <= n; i += AVX_VECTOR_SIZE) {
            __m256 uSimd = MM256_LOAD(U + i);
            __m256 wSimd = MM256_LOAD(W + i);
            accumulateSum += mm256_ipow_sum(_mm256_fmadd_ps(wSimd, uSimd, aSimd), k);
            accumulateDiv += mm256_horizontal_add(wSimd);
        }
        if (i < n) {
            __m256i mask = _mm256_setr_epi32(-(i < n), -((i + 1) < n), -((i + 2) < n), -((i + 3) < n),
                                             -((i + 4) < n), -((i + 5) < n), -((i + 6) < n), -((i + 7) < n));
            __m256 uSimd = _mm256_maskload_ps(U + i, mask);
            __m256 wSimd = _mm256_maskload_ps(W + i, mask);
            accumulateSum += mm256_ipow_sum_partial(_mm256_fmadd_ps(wSimd, uSimd, aSimd), k, n - i);
            accumulateDiv += mm256_horizontal_add(wSimd);
        }
        return accumulateSum / accumulateDiv;
    }


    float gm_parrallel(const float *U, const float *W, float a, int k, unsigned int n, unsigned int nthread) {
        float accumulateSum = 0, accumulateDiv = 0;
        std::mutex mutex;
        auto lambda = [&mutex, &accumulateSum, &accumulateDiv, &U, &W, a, k](unsigned int start, unsigned int end) {
            float accumulateSumLocal = 0, accumulateDivLocal = 0;
            __m256 aSimd = _mm256_set1_ps(-a);
            unsigned int i;
            for (i = start; i + AVX_VECTOR_SIZE <= end; i += AVX_VECTOR_SIZE) {
                __m256 uSimd = MM256_LOAD(U + i);
                __m256 wSimd = MM256_LOAD(W + i);
                accumulateSumLocal += mm256_ipow_sum(_mm256_fmadd_ps(wSimd, uSimd, aSimd), k);
                accumulateDivLocal += mm256_horizontal_add(wSimd);
            }
            if (i < end) {
                __m256i mask = _mm256_setr_epi32(-(i < end), -((i + 1) < end), -((i + 2) < end), -((i + 3) < end),
                                                 -((i + 4) < end), -((i + 5) < end), -((i + 6) < end),
                                                 -((i + 7) < end));

                __m256 uSimd = _mm256_maskload_ps(U + i, mask);
                __m256 wSimd = _mm256_maskload_ps(W + i, mask);
                accumulateSumLocal += mm256_ipow_sum_partial(_mm256_fmadd_ps(wSimd, uSimd, aSimd), k, end - i);
                accumulateDivLocal += mm256_horizontal_add(wSimd);
            }

            std::lock_guard<std::mutex> lock(mutex);
            accumulateSum += accumulateSumLocal;
            accumulateDiv += accumulateDivLocal;
        };

        parallel_func(0u, n, lambda, nthread);

        return accumulateSum / accumulateDiv;
    }
}