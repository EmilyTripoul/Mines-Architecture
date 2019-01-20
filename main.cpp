#include <iostream>
#include <chrono>
#include <ctime>
#include <random>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <functional>
#include <cmath>
#include <thread>
#include <omp.h>
#include <cstdlib>

#include "cmd_parser.h"
#include "sequential.h"
#include "vectorial.h"
#include "parallel.h"
#include "utils.h"

#ifdef CUDA_AVAILABLE
    #include "gpu.h"
#endif

using namespace utils;
using namespace std::placeholders;

static double t_duration_avg_ref = 0;

template<typename functionHandle>
void run_benchmarck(const std::string &benchmarkName, const functionHandle &f, const benchmarkParams &params,
                    unsigned int number_run = 5) {
    typedef std::chrono::high_resolution_clock t_clock;

    double c_duration_min = std::numeric_limits<double>::max(), c_duration_max = 0, c_duration_avg = 0;
    double t_duration_min = std::numeric_limits<double>::max(), t_duration_max = 0, t_duration_avg = 0;
    float previousResult = 0;

    for (int i = 0; i < number_run; i++) {
        std::clock_t c_start = std::clock();
        auto t_start = std::chrono::high_resolution_clock::now();

        auto result = f(params);

        std::clock_t c_end = std::clock();
        auto t_end = std::chrono::high_resolution_clock::now();

        auto c_duration = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
        auto t_duration = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        if (c_duration < c_duration_min) c_duration_min = c_duration;
        if (t_duration < t_duration_min) t_duration_min = t_duration;
        if (c_duration > c_duration_max) c_duration_max = c_duration;
        if (t_duration > t_duration_max) t_duration_max = t_duration;
        c_duration_avg += c_duration;
        t_duration_avg += t_duration;

        if (i > 0 && abs(result - previousResult) > 1e-5)
            std::cerr << "Result is not equal to previous one : " << previousResult << " | " << result << std::endl;
        previousResult = result;
    }
    c_duration_avg /= number_run;
    t_duration_avg /= number_run;

    if (benchmarkName == std::string("ipow")) t_duration_avg_ref = t_duration_avg;

    std::cout << benchmarkName << "\t\t"
              << std::fixed << std::setprecision(2)
              << c_duration_min << "\t" << c_duration_max << "\t" << c_duration_avg << "\t"
              << t_duration_min << "\t" << t_duration_max << "\t" << t_duration_avg << " ("
              << 100 * (t_duration_avg_ref / t_duration_avg - 1) << "%)" << "\t"
              << previousResult << std::endl;

}

#define RUN_BENCHMARK_FUNC(NAME, FUNC)  run_benchmarck(std::string(NAME), FUNC, params, number_run)

void run_all(const benchmarkParams &params, unsigned int number_run = 5) {

    std::cout << "Benchmark" << "\t" << "CPU time (ms)" << "\t\t" << "Wall time (ms)\tResult" << std::endl;
    std::cout << "\t\t" << "min\tmax\tavg\t" << "min\tmax\tavg\t" << std::endl;
    std::cout << "=============================================================================" << std::endl;

    std::cout << "SEQUENTIAL" << std::endl;
    RUN_BENCHMARK_SEQUENTIAL(RUN_BENCHMARK_FUNC)
    std::cout << "-----------------------------------------------------------------------------" << std::endl;
    std::cout << "VECTORIAL" << std::endl;
    RUN_BENCHMARK_VECTORIAL(RUN_BENCHMARK_FUNC)
    std::cout << "-----------------------------------------------------------------------------" << std::endl;
    std::cout << "PARALLEL" << std::endl;
    RUN_BENCHMARK_PARALLEL(RUN_BENCHMARK_FUNC)
#ifdef CUDA_AVAILABLE
    std::cout << "-----------------------------------------------------------------------------" << std::endl;
    std::cout << "CUDA" << std::endl;
    RUN_BENCHMARK_GPU(RUN_BENCHMARK_FUNC)
#endif
    std::cout << "=============================================================================" << std::endl;

}

template<class Generator>
void init_random_vector(float *v, unsigned int start, unsigned int end, const Generator &g) {
    std::generate(v + start, v + end, g);
}

int main(int argc, char **argv) {
    CmdParser cmdParser(argc, argv);
    if (cmdParser.optionExists("-h") || cmdParser.getPositionalNumber() < 1) {
        std::cout
                << "Usage : ./bench n [k] [a] [-h] [-run <int>]\n\tn : unsigned int > 0\n\tk : int (default = 1)\n\ta : float (default = 0)\n\t-h : help\n\t-run <int> : run number (default = 10)\n\t-nth <int> : thread number (default = 0)"
                << std::endl;
        return 0;
    }

    unsigned int runNumber = (cmdParser.optionExists("-run")) ? cmdParser.getOptionAs<unsigned int>("-run") : 10;
    unsigned int threadNumberMax;
    if (cmdParser.optionExists("-nth")) {
        threadNumberMax = cmdParser.getOptionAs<unsigned int>("-nth");
    } else {
        const static unsigned nb_threads_hint = std::thread::hardware_concurrency();
        threadNumberMax = (nb_threads_hint == 0u) ? 8u : nb_threads_hint;
    }
    omp_set_num_threads(threadNumberMax);

    unsigned int n = cmdParser.getPositionalAs<unsigned int>(0);
    int k = (cmdParser.getPositionalNumber() >= 2) ? cmdParser.getPositionalAs<int>(1) : 1;
    float a = (cmdParser.getPositionalNumber() >= 3) ? cmdParser.getPositionalAs<float>(2) : 0;
    benchmarkParams params{
        nullptr, nullptr,
#ifdef CUDA_AVAILABLE
        nullptr, nullptr,
#endif
        a, k, n, threadNumberMax,utils::E_UNDEF};

    std::cout << "n=" << n << "\t" << "k=" << k << "\t" << "a=" << a << std::endl;

#ifdef FORCE_ALIGN
    params.U = static_cast<float*>(std::aligned_alloc(256, n*sizeof(float)));
    params.W = static_cast<float*>(std::aligned_alloc(256, n*sizeof(float)));
#else
    params.U = new float[n];
    params.W = new float[n];
#endif

#pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        int threadNumber = omp_get_num_threads();
        unsigned int start = (threadId) * n / threadNumber;
        unsigned int end = (threadId + 1) * n / threadNumber;

        std::random_device rd;
        std::uniform_real_distribution<double> distribution(0, 1);
        std::mt19937 engine(rd());
        auto generator = std::bind(distribution, engine);
        init_random_vector(params.U, start, end, generator);
        init_random_vector(params.W, start, end, generator);
    };

#ifdef CUDA_AVAILABLE
    gpu::initParams(params);
#endif


    run_all(params, runNumber);


#ifdef FORCE_ALIGN
    std::free(params.U);
    std::free(params.W);
#else
    delete[] params.U;
    delete[] params.W;
#endif

#ifdef CUDA_AVAILABLE
    gpu::freeParams(params);
#endif

    return 0;
}

