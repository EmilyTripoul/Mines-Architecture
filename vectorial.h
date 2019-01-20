//
// Created by Emily & Marc on 10/12/2018.
//
#pragma once

#include "utils.h"
#include <immintrin.h>

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


    // ******************************
    // VECTORIAL TOOLS
    // ******************************

#define AVX_VECTOR_SIZE 8


    inline void show_mm256(__m256 &x) {
        float *res = (float *) &(x);
        std::cout << res[0] << "\t" << res[1] << "\t" << res[2] << "\t" << res[3] << "\t" << res[4] << "\t" << res[5]
                  << "\t" << res[6] << "\t" << res[7] << std::endl;
    }

    inline __m128 mm256_get_low(const __m256 &x) {
        return _mm256_castps256_ps128(x);
    }

    inline __m128 mm256_get_high(const __m256 &x) {
        return _mm256_extractf128_ps(x, 1);
    }

    inline float mm128_horizontal_add(const __m128 &x) {
        __m128 shuff1 = _mm_movehdup_ps(x);                   // duplicate elements 3,1 -> [ 3 3 1 1 ]
        __m128 sum2 = _mm_add_ps(x, shuff1);
        __m128 shuff2 = _mm_movehl_ps(sum2, sum2);                  // same,same doesn't waste a movaps with AVX
        __m128 sum3 = _mm_add_ss(sum2, shuff2);
        return _mm_cvtss_f32(sum3); // SSE3 issue ?
    }

    inline float mm128_horizontal_add_shuffle(const __m128 &x) {
        __m128 shuff1 = _mm_movehdup_ps(x);                   // duplicate elements 3,1 -> [ 3 3 1 1 ]
        __m128 sum2 = _mm_add_ps(x, shuff1);
        __m128 shuff2 = _mm_movehl_ps(sum2, sum2);                  // same,same doesn't waste a movaps with AVX
        __m128 sum3 = _mm_add_ps(sum2, shuff2);
        __m128 shuff3 = _mm_shuffle_ps(sum3, sum3, 0x1);
        __m128 sum4 = _mm_add_ss(sum3, shuff3);
        return _mm_cvtss_f32(sum4); // SSE3 issue ?
    }

    inline float mm256_horizontal_add(const __m256 &x) {
        __m128 sum1 = _mm_add_ps(mm256_get_low(x), mm256_get_high(x));
        return mm128_horizontal_add(sum1); // SSE3 issue ?
    }

    inline __m256 mm256_ipow(__m256 base, int exp) {
        __m256 result = _mm256_set1_ps(1.f);
        while (true) {
            if (exp & 1) result = _mm256_mul_ps(result, base);
            exp >>= 1;
            if (!exp) break;
            base = _mm256_mul_ps(base, base);
        }
        return result;
    }

    inline float mm256_ipow_sum(__m256 base, int exp) {
        if (exp > 0) return mm256_horizontal_add(mm256_ipow(base, exp));
        else if (exp == 0) return AVX_VECTOR_SIZE;
        else return mm256_horizontal_add(_mm256_rcp_ps(mm256_ipow(base, -exp)));
    }

    inline float mm256_ipow_sum_partial(__m256 base, int exp, unsigned int size = AVX_VECTOR_SIZE) {
        if (exp > 0) return mm256_horizontal_add(mm256_ipow(base, exp));
        else if (exp == 0) return static_cast<float>(size);
        else {
            __m256 mask = _mm256_setr_ps(0 < size, 1 < size, 2 < size, 3 < size, 4 < size, 5 < size, 6 < size,
                                         7 < size);
            __m256 maski = _mm256_setr_ps(0 >= size, 1 >= size, 2 >= size, 3 >= size, 4 >= size, 5 >= size, 6 >= size,
                                          7 >= size);
            return mm256_horizontal_add(
                    _mm256_mul_ps(_mm256_rcp_ps(_mm256_add_ps(mm256_ipow(base, -exp), maski)), mask));
        }
    }
}
