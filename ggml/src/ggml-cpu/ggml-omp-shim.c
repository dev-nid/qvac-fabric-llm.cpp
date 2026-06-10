#ifndef GGML_OMP_SHIM_BUILD
#define GGML_OMP_SHIM_BUILD
#endif
#include "ggml-omp-shim.h"
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

static volatile int ggml_omp_shim_first_call = 1;

GGML_OMP_SHIM_API
void ggml_omp_shim_parallel(ggml_omp_parallel_cb_t callback, void * data, int n_threads) {
    #pragma omp parallel num_threads(n_threads)
    {
        int nth = omp_get_num_threads();
        int ith = omp_get_thread_num();
        if (ggml_omp_shim_first_call && ith == 0) {
            ggml_omp_shim_first_call = 0;
            fprintf(stderr, "GGML_OMP_SHIM: parallel region active — requested %d threads, got %d threads (max=%d)\n",
                    n_threads, nth, omp_get_max_threads());
        }
        callback(data, ith, nth);
    }
}

GGML_OMP_SHIM_API
void ggml_omp_shim_barrier(void) {
    #pragma omp barrier
}

GGML_OMP_SHIM_API
void ggml_omp_shim_init_env(void) {
    if (!getenv("KMP_BLOCKTIME")) {
#ifdef _WIN32
        _putenv_s("KMP_BLOCKTIME", "200");
#else
        setenv("KMP_BLOCKTIME", "200", 0);
#endif
    }
}
