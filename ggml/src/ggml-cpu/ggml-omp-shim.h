#ifndef GGML_OMP_SHIM_H
#define GGML_OMP_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef GGML_OMP_SHIM_BUILD
#    define GGML_OMP_SHIM_API __declspec(dllexport)
#  else
#    define GGML_OMP_SHIM_API __declspec(dllimport)
#  endif
#else
#  define GGML_OMP_SHIM_API __attribute__((visibility("default")))
#endif

typedef void (*ggml_omp_parallel_cb_t)(void * data, int ith, int nth);

GGML_OMP_SHIM_API void ggml_omp_shim_parallel(ggml_omp_parallel_cb_t callback, void * data, int n_threads);
GGML_OMP_SHIM_API void ggml_omp_shim_barrier(void);
GGML_OMP_SHIM_API void ggml_omp_shim_init_env(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_OMP_SHIM_H */
