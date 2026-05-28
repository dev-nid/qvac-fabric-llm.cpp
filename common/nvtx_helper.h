// Lightweight NVTX scoped range helper, used for nsys-trace attribution of
// the DFlash speculative loop phases (encoder / draft_decode / target_decode).
//
// Markers are no-ops unless the build defines LLAMA_NVTX_ENABLED *and* the
// process is launched under nsys with --trace=nvtx. Outside of profiling
// they cost a few ns per push/pop. Intentionally hand-rolled rather than
// pulling in cuda toolkit's nvtx3 headers (which require CUDA 12.5+).

#pragma once

#ifdef LLAMA_NVTX_ENABLED
#include <nvToolsExt.h>

class nvtx_scoped_range {
public:
    explicit nvtx_scoped_range(const char * name) {
        nvtxRangePushA(name);
    }
    ~nvtx_scoped_range() {
        nvtxRangePop();
    }
    nvtx_scoped_range(const nvtx_scoped_range &) = delete;
    nvtx_scoped_range & operator=(const nvtx_scoped_range &) = delete;
};

#define NVTX_RANGE(name)                                              \
    nvtx_scoped_range _nvtx_range_##__LINE__##__(name)
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()

#else  // LLAMA_NVTX_ENABLED

#define NVTX_RANGE(name) do { (void) (name); } while (0)
#define NVTX_PUSH(name)  do { (void) (name); } while (0)
#define NVTX_POP()       do {} while (0)

#endif  // LLAMA_NVTX_ENABLED
