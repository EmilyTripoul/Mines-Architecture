//
// Created by Emily & Marc on 10/12/2018.
//
#pragma once

#include <immintrin.h>
#include <numeric>
#include <algorithm>
#include <vector>
#include <thread>
#include <atomic>

#include <iostream>

namespace utils {

    typedef enum runMode {
        E_UNDEF = 0,
        E_SEQUENTIAL,
        E_PARALLEL,
        E_VECTORIAL
    } runMode;

    typedef struct benchmarkParams {
        float *U;
        float *W;
#ifdef CUDA_AVAILABLE
        float *U_gpu;
        float *W_gpu;
#endif
        float a;
        int k;
        unsigned int n;
        unsigned int nthread;
        runMode mode;
    } benchmarkParams;


    template<typename functionHandle>
    float wrapper_seq(const functionHandle &f, const utils::benchmarkParams &params) {
        return f(params.U, params.W, params.a, params.k, params.n);
    }

    template<typename functionHandle>
    float wrapper_par(const functionHandle &f, const utils::benchmarkParams &params) {
        return f(params.U, params.W, params.a, params.k, params.n, params.nthread);
    }

    template<typename functionHandle, typename functionHandleVect>
    float
    wrapper_par_mode(const functionHandle &f, const functionHandleVect &fVect, const utils::benchmarkParams &params) {
        if (params.mode == E_SEQUENTIAL || params.mode == E_PARALLEL) {
            return f(params.U, params.W, params.a, params.k, params.n, params.nthread);
        } else {
            return fVect(params.U, params.W, params.a, params.k, params.n, params.nthread);
        }
    }


#define UTILS_WRAP(WRAPPER, FUNCTION) [](const utils::benchmarkParams &params){return WRAPPER(FUNCTION, params);}


    // ******************************
    // SEQUENTIAL TOOLS
    // ******************************
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

    inline float ipowf(float base, int exp) {
        if (exp > 0) return ipowf_helper(base, exp);
        else if (exp == 0) return 1.f;
        else return 1.f / ipowf_helper(base, -exp);
    }

    // ******************************
    // PARALLEL TOOLS
    // ******************************

    template<typename Index, typename Callable>
    void parallel_func(Index start, Index end, Callable func, unsigned int nthread) {
        // Size of a slice for the range functions
        Index n = end - start + 1;
        Index slice = (Index) std::round(n / static_cast<double> (nthread));
        slice = std::max(slice, Index(1));

        // Create pool and launch jobs
        std::vector<std::thread> pool;
        pool.reserve(nthread);
        Index i1 = start;
        Index i2 = std::min(start + slice, end);
        for (unsigned i = 0; i + 1 < nthread && i1 < end; ++i) {
            pool.emplace_back(func, i1, i2);
            i1 = i2;
            i2 = std::min(i2 + slice, end);
        }
        if (i1 < end) {
            pool.emplace_back(func, i1, end);
        }

        // Wait for jobs to finish
        for (std::thread &t : pool) {
            if (t.joinable()) {
                t.join();
            }
        }

    }

    template<typename BinaryOp>
    float fetch_and_op(std::atomic<float> &atomic, const float value, BinaryOp op = BinaryOp{}) {
        float old = atomic;
        float desired;
        do {
            desired = op(old, value);
        } while (!atomic.compare_exchange_weak(old, desired));
        return desired;
    }

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
        return _mm_cvtss_f32(sum3); // SSE3 issue ?
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
