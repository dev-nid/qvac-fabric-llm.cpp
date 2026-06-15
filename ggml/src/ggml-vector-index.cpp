// ggml-vector-index.cpp — POC scalar implementation of the fabric vector
// index C API declared in `ggml/include/ggml-vector-index.h`.
//
// Storage: full f32 vectors as a contiguous std::vector<float>. ID map uses
// std::unordered_map<uint64_t, size_t> for lookup and a parallel vector for
// the slot->id reverse map. Remove uses swap-with-last.
//
// Search: naive scalar dot product across all slots + min-heap of size k.
// No SIMD, no GPU. Correctness over speed; the optimization phase will swap
// the storage layout and kernel without touching the C API.

#include "ggml-vector-index.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint8_t  kTvimMagic[4]   = { 'T', 'V', 'P', 'I' };
constexpr uint8_t  kTvimVersion    = 1;
constexpr size_t   kTvimHeaderSize = 16;

// Top-k via min-heap of (score, id). The heap holds at most `k` candidates;
// each new score is compared against the smallest in the heap.
struct ScoreId {
    float    score;
    uint64_t id;
};

struct MinHeapCmp {
    bool operator()(const ScoreId & a, const ScoreId & b) const {
        // Min-heap by score (smallest score at the top).
        return a.score > b.score;
    }
};

} // namespace

// Lifetime-managed instance state. Lives behind the opaque
// `ggml_vec_index_t` typedef.
struct ggml_vec_index {
    int dim       = 0;
    int bit_width = 32;

    // Flat row-major storage: `data[slot * dim + i]` is component i of vec slot.
    std::vector<float> data;

    // slot -> external id (parallel to logical slot index).
    std::vector<uint64_t> slot_to_id;

    // external id -> slot.
    std::unordered_map<uint64_t, size_t> id_to_slot;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width) {
    if (dim <= 0) {
        return nullptr;
    }
    if (bit_width <= 0 || bit_width > 32) {
        // POC ignores bit_width but still validates the range so callers
        // surface bad config early.
        return nullptr;
    }
    auto * idx = new (std::nothrow) ggml_vec_index();
    if (idx == nullptr) {
        return nullptr;
    }
    idx->dim       = dim;
    idx->bit_width = bit_width;
    return idx;
}

void ggml_vec_index_free(ggml_vec_index_t * idx) {
    delete idx;
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

int ggml_vec_index_add(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids) {

    if (idx == nullptr || vectors == nullptr || ids == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n < 0) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n == 0) {
        return GGML_VEC_INDEX_OK;
    }

    // Atomic add: detect duplicates first (against existing AND in-batch),
    // bail before mutating any state.
    for (int i = 0; i < n; ++i) {
        if (idx->id_to_slot.find(ids[i]) != idx->id_to_slot.end()) {
            return GGML_VEC_INDEX_E_DUPLICATE;
        }
        for (int j = i + 1; j < n; ++j) {
            if (ids[i] == ids[j]) {
                return GGML_VEC_INDEX_E_DUPLICATE;
            }
        }
    }

    const size_t base_slot = idx->slot_to_id.size();
    const size_t dim_sz    = static_cast<size_t>(idx->dim);

    try {
        idx->data.resize((base_slot + static_cast<size_t>(n)) * dim_sz);
        idx->slot_to_id.resize(base_slot + static_cast<size_t>(n));
        idx->id_to_slot.reserve(base_slot + static_cast<size_t>(n));
    } catch (const std::bad_alloc &) {
        idx->data.resize(base_slot * dim_sz);
        idx->slot_to_id.resize(base_slot);
        return GGML_VEC_INDEX_E_OOM;
    }

    for (int i = 0; i < n; ++i) {
        const size_t slot = base_slot + static_cast<size_t>(i);
        std::memcpy(
            idx->data.data() + slot * dim_sz,
            vectors + static_cast<size_t>(i) * dim_sz,
            dim_sz * sizeof(float));
        idx->slot_to_id[slot] = ids[i];
        idx->id_to_slot.emplace(ids[i], slot);
    }
    return GGML_VEC_INDEX_OK;
}

int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    auto it = idx->id_to_slot.find(id);
    if (it == idx->id_to_slot.end()) {
        return 0;
    }
    const size_t slot     = it->second;
    const size_t last     = idx->slot_to_id.size() - 1;
    const size_t dim_sz   = static_cast<size_t>(idx->dim);

    if (slot != last) {
        // Move last vector into the freed slot and update its id mapping.
        std::memcpy(
            idx->data.data() + slot * dim_sz,
            idx->data.data() + last * dim_sz,
            dim_sz * sizeof(float));
        const uint64_t moved_id = idx->slot_to_id[last];
        idx->slot_to_id[slot]   = moved_id;
        idx->id_to_slot[moved_id] = slot;
    }

    idx->slot_to_id.pop_back();
    idx->data.resize(last * dim_sz);
    idx->id_to_slot.erase(it);
    return 1;
}

int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id) {
    if (idx == nullptr) {
        return 0;
    }
    return idx->id_to_slot.count(id) != 0 ? 1 : 0;
}

void ggml_vec_index_prepare(ggml_vec_index_t * /*idx*/) {
    // POC no-op. Future: warm caches, materialize codebooks, etc.
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

namespace {

// Scalar dot product of two `dim`-length f32 vectors.
inline float dot(const float * a, const float * b, int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

// Run a single query against all slots, write top-k into out_scores/out_ids.
// If the index holds fewer than k entries, pad with sentinels.
void search_one(
    const ggml_vec_index_t & idx,
    const float            * query,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids) {

    const int    dim     = idx.dim;
    const size_t n_slots = idx.slot_to_id.size();

    std::priority_queue<ScoreId, std::vector<ScoreId>, MinHeapCmp> heap;

    for (size_t slot = 0; slot < n_slots; ++slot) {
        const float s = dot(
            query, idx.data.data() + slot * static_cast<size_t>(dim), dim);
        if (heap.size() < static_cast<size_t>(k)) {
            heap.push({ s, idx.slot_to_id[slot] });
        } else if (s > heap.top().score) {
            heap.pop();
            heap.push({ s, idx.slot_to_id[slot] });
        }
    }

    // Drain the heap into a temporary descending list.
    std::vector<ScoreId> drained;
    drained.reserve(heap.size());
    while (!heap.empty()) {
        drained.push_back(heap.top());
        heap.pop();
    }
    std::reverse(drained.begin(), drained.end()); // now descending by score

    for (int i = 0; i < k; ++i) {
        if (static_cast<size_t>(i) < drained.size()) {
            out_scores[i] = drained[i].score;
            out_ids[i]    = drained[i].id;
        } else {
            out_scores[i] = -FLT_MAX;
            out_ids[i]    = UINT64_MAX;
        }
    }
}

} // namespace

int ggml_vec_index_search(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids) {

    if (idx == nullptr || queries == nullptr ||
        out_scores == nullptr || out_ids == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q < 0 || k <= 0) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q == 0) {
        return GGML_VEC_INDEX_OK;
    }

    const int dim = idx->dim;
    for (int q = 0; q < n_q; ++q) {
        search_one(
            *idx,
            queries + static_cast<size_t>(q) * static_cast<size_t>(dim),
            k,
            out_scores + static_cast<size_t>(q) * static_cast<size_t>(k),
            out_ids    + static_cast<size_t>(q) * static_cast<size_t>(k));
    }
    return GGML_VEC_INDEX_OK;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

int ggml_vec_index_write(ggml_vec_index_t * idx, const char * path) {
    if (idx == nullptr || path == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
        return GGML_VEC_INDEX_E_IO;
    }

    // Header: 16 bytes. Layout matches the comment block in the header file.
    uint8_t header[kTvimHeaderSize] = {};
    std::memcpy(header, kTvimMagic, 4);
    header[4] = kTvimVersion;
    header[5] = static_cast<uint8_t>(idx->bit_width);
    header[6] = 0;
    header[7] = 0;
    const uint32_t dim_le = static_cast<uint32_t>(idx->dim);
    const uint32_t n_le   = static_cast<uint32_t>(idx->slot_to_id.size());
    std::memcpy(header + 8, &dim_le, 4);
    std::memcpy(header + 12, &n_le, 4);

    f.write(reinterpret_cast<const char *>(header), sizeof(header));
    if (!f) { return GGML_VEC_INDEX_E_IO; }

    if (n_le > 0) {
        f.write(
            reinterpret_cast<const char *>(idx->data.data()),
            static_cast<std::streamsize>(idx->data.size() * sizeof(float)));
        if (!f) { return GGML_VEC_INDEX_E_IO; }

        f.write(
            reinterpret_cast<const char *>(idx->slot_to_id.data()),
            static_cast<std::streamsize>(idx->slot_to_id.size() * sizeof(uint64_t)));
        if (!f) { return GGML_VEC_INDEX_E_IO; }
    }

    f.flush();
    if (!f) { return GGML_VEC_INDEX_E_IO; }
    return GGML_VEC_INDEX_OK;
}

ggml_vec_index_t * ggml_vec_index_load(const char * path) {
    if (path == nullptr) {
        return nullptr;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return nullptr;
    }

    uint8_t header[kTvimHeaderSize] = {};
    f.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!f || f.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        return nullptr;
    }
    if (std::memcmp(header, kTvimMagic, 4) != 0) {
        return nullptr;
    }
    if (header[4] != kTvimVersion) {
        return nullptr;
    }

    const int bit_width = static_cast<int>(header[5]);
    uint32_t dim_le = 0;
    uint32_t n_le   = 0;
    std::memcpy(&dim_le, header + 8, 4);
    std::memcpy(&n_le,   header + 12, 4);
    if (dim_le == 0) {
        return nullptr;
    }
    const int dim = static_cast<int>(dim_le);

    auto * idx = ggml_vec_index_create(dim, bit_width);
    if (idx == nullptr) {
        return nullptr;
    }
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t n      = static_cast<size_t>(n_le);

    try {
        idx->data.resize(n * dim_sz);
        idx->slot_to_id.resize(n);
        idx->id_to_slot.reserve(n);
    } catch (const std::bad_alloc &) {
        ggml_vec_index_free(idx);
        return nullptr;
    }

    if (n > 0) {
        f.read(
            reinterpret_cast<char *>(idx->data.data()),
            static_cast<std::streamsize>(n * dim_sz * sizeof(float)));
        if (!f) {
            ggml_vec_index_free(idx);
            return nullptr;
        }

        f.read(
            reinterpret_cast<char *>(idx->slot_to_id.data()),
            static_cast<std::streamsize>(n * sizeof(uint64_t)));
        if (!f) {
            ggml_vec_index_free(idx);
            return nullptr;
        }

        for (size_t slot = 0; slot < n; ++slot) {
            const uint64_t id = idx->slot_to_id[slot];
            auto [it, inserted] = idx->id_to_slot.emplace(id, slot);
            if (!inserted) {
                // Duplicate id in persisted file: corrupted.
                ggml_vec_index_free(idx);
                return nullptr;
            }
        }
    }

    return idx;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

int ggml_vec_index_len(const ggml_vec_index_t * idx) {
    return idx ? static_cast<int>(idx->slot_to_id.size()) : 0;
}

int ggml_vec_index_dim(const ggml_vec_index_t * idx) {
    return idx ? idx->dim : 0;
}

int ggml_vec_index_bit_width(const ggml_vec_index_t * idx) {
    return idx ? idx->bit_width : 0;
}
