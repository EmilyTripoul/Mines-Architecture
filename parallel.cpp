//
// Created by Emily & Marc on 10/12/2018.
//

#include "parallel.h"
#include <omp.h>
#include "utils.h"
using namespace utils;

namespace parallel {

    float gm_omp(const float *U, const float *V, float a, int k, unsigned int n, unsigned int nthread) {
        float accumulateSum = 0, accumulateDiv = 0;
#pragma omp parallel for reduction(+:accumulateSum,accumulateDiv) schedule(static) num_threads(nthread)
        for (int i = 0; i < static_cast<int>(n); i++) {
            accumulateSum += ipowf(V[i] * U[i] - a, k);
            accumulateDiv += V[i];
        }
        return accumulateSum / accumulateDiv;
    }
/*
    float gm_std(const float *U, const float *V, float a, int k, unsigned int n) {
        float accumulateSum = 0, accumulateDiv = 0;
        std::reduce(std::execution::par_unseq, 0, n, 0, [a,k,n](const auto &i, const auto &b) { return 0;});
        template<class ExecutionPolicy, class ForwardIt, class T, class BinaryOp>

        T reduce(ExecutionPolicy&& policy,
                 ForwardIt first, ForwardIt last, T init, BinaryOp binary_op);
#pragma omp parallel for reduction(+:accumulateSum,accumulateDiv) schedule(static)
        for (int i = 0; i < n; i++) {
            accumulateSum += ipowf(V[i] * U[i] - a, k);
            accumulateDiv += V[i];
        }
        return accumulateSum / accumulateDiv;
    }*/

    float gm_std_thread(const float *U, const float *V, float a, int k, unsigned int n, unsigned int nthread) {
        std::mutex mutex;
        float accumulateSum = 0, accumulateDiv = 0;
        auto lambda = [&mutex, &accumulateSum, &accumulateDiv, &U, &V, a,k](unsigned int start, unsigned int end){
            float accumulateSumLocal = 0, accumulateDivLocal = 0;
            for (unsigned int i = start; i < end; i++) {
                accumulateSumLocal += ipowf(V[i] * U[i] - a, k);
                accumulateDivLocal += V[i];
            }

            std::lock_guard<std::mutex> lock(mutex);
            accumulateSum += accumulateSumLocal;
            accumulateDiv += accumulateDivLocal;
        };

        parallel_func(0u,n, lambda, nthread);

        return accumulateSum / accumulateDiv;

    }

    float gm_std_thread_atomic(const float *U, const float *V, float a, int k, unsigned int n, unsigned int nthread) {
        std::atomic<float> accumulateSum = 0, accumulateDiv = 0;
        auto lambda = [&accumulateSum, &accumulateDiv, &U, &V, a,k](unsigned int start, unsigned int end){
            float accumulateSumLocal = 0, accumulateDivLocal = 0;
            for (unsigned int i = start; i < end; i++) {
                accumulateSumLocal += ipowf(V[i] * U[i] - a, k);
                accumulateDivLocal += V[i];
            }

            fetch_and_op<std::plus<> >(accumulateSum, accumulateSumLocal);
            fetch_and_op<std::plus<> >(accumulateDiv, accumulateDivLocal);
        };

        parallel_func(0u,n, lambda, nthread);

        return accumulateSum / accumulateDiv;

    }


}