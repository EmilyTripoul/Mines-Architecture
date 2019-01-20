//
// Created by Emily & Marc on 10/12/2018.
//
#pragma once

#include <numeric>
#include <algorithm>
#include <vector>
#include <thread>
#include <atomic>
#include <cmath>

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


}
