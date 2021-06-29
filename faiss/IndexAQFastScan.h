/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// #include <faiss/IndexAQ.h>
#include <faiss/impl/AdditiveQuantizer.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

/** Fast scan version of IndexAQ. Works for 4-bit AQ for now.
 *
 * The codes are not stored sequentially but grouped in blocks of size bbs.
 * This makes it possible to compute distances quickly with SIMD instructions.
 *
 * Implementations:
 * 12: blocked loop with internal loop on Q with qbs
 * 13: same with reservoir accumulator to store results
 * 14: no qbs with heap accumulator
 * 15: no qbs with reservoir accumulator
 */

struct IndexAQFastScan : Index {
    AdditiveQuantizer* aq;

    // quantizer for norms
    AdditiveQuantizer* norm_aq;

    // implementation to select
    int implem = 0;
    // skip some parts of the computation (for timing)
    int skip = 0;

    // size of the kernel
    int bbs;     // set at build time
    int qbs = 0; // query block size 0 = use default

    size_t M;
    size_t nbits;
    size_t ksub;
    size_t code_size;

    // packed version of the codes
    size_t ntotal2;
    size_t M2;

    float mean_norm = 0;

    AlignedTable<uint8_t> codes;

    // this is for testing purposes only (set when initialized by IndexAQ)
    std::vector<uint8_t> orig_codes;

    size_t chunk_size = 10000;

    IndexAQFastScan(
            AdditiveQuantizer* aq,
            AdditiveQuantizer* norm_aq,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexAQFastScan(
            AdditiveQuantizer* aq,
            MetricType metric = METRIC_INNER_PRODUCT,
            int bbs = 32);

    IndexAQFastScan();

    ~IndexAQFastScan();

    /// build from an existing IndexAQ
    // explicit IndexAQFastScan(const IndexAQ& orig, int bbs = 32);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void reset() override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    // called by search function
    void compute_quantized_LUT(
            idx_t n,
            const float* x,
            uint8_t* lut,
            float* normalizers) const;

    template <bool is_max>
    void search_dispatch_implem(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    template <class C>
    void search_implem_2(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    template <class C>
    void search_implem_12(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl) const;

    template <class C>
    void search_implem_14(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl) const;

    void compute_codes(uint8_t* tmp_codes, idx_t n, const float* x) const;

    void compute_LUT(float* lut, idx_t n, const float* x) const;
};

struct AQFastScanStats {
    uint64_t t0, t1, t2, t3;
    AQFastScanStats() {
        reset();
    }
    void reset() {
        memset(this, 0, sizeof(*this));
    }
};

FAISS_API extern AQFastScanStats AQFastScan_stats;

} // namespace faiss
