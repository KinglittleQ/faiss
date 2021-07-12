/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexAQFastScan.h>

#include <limits.h>
#include <cassert>
#include <memory>

#include <omp.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/utils/quantize_lut.h>

namespace faiss {

using namespace simd_result_handlers;

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

IndexAQFastScan::IndexAQFastScan(
        AdditiveQuantizer* aq,
        AdditiveQuantizer* norm_aq,
        MetricType metric,
        int bbs)
        : Index(aq->d, metric),
          aq(aq),
          norm_aq(norm_aq),
          bbs(bbs),
          nbits(aq->nbits[0]),
          ksub(1 << nbits),
          ntotal2(0) {
    FAISS_THROW_IF_NOT(aq->nbits[0] == 4);
    FAISS_THROW_IF_NOT(!(metric_type == METRIC_L2 && norm_aq == nullptr));

    if (metric_type == METRIC_L2) {
        FAISS_THROW_IF_NOT(norm_aq->nbits[0] == 4);
        M = aq->M + norm_aq->M;
        code_size = (M * nbits + 7) / 8;
    } else {
        M = aq->M;
        code_size = aq->code_size;
    }
    M2 = roundup(M, 2);

    is_trained = false;
}

IndexAQFastScan::IndexAQFastScan(
        AdditiveQuantizer* aq,
        MetricType metric,
        int bbs)
        : Index(aq->d, metric),
          aq(aq),
          norm_aq(nullptr),
          bbs(bbs),
          M(aq->M),
          nbits(aq->nbits[0]),
          ksub(1 << nbits),
          code_size(aq->code_size),
          ntotal2(0),
          M2(roundup(M, 2)) {
    FAISS_THROW_IF_NOT(aq->nbits[0] == 4);
    FAISS_THROW_IF_NOT(metric_type == METRIC_INNER_PRODUCT);

    is_trained = false;
}

IndexAQFastScan::IndexAQFastScan() : bbs(0), ntotal2(0), M2(0) {}

IndexAQFastScan::~IndexAQFastScan() {}

// IndexAQFastScan::IndexAQFastScan(const IndexAQ& orig, int bbs)
//         : Index(orig.d, orig.metric_type), aq(orig.aq), bbs(bbs) {
//     FAISS_THROW_IF_NOT(orig.aq->nbits == 4);
//     ntotal = orig.ntotal;
//     is_trained = orig.is_trained;
//     orig_codes = orig.codes.data();

//     qbs = 0; // means use default

//     // pack the codes

//     size_t M = aq->M;

//     FAISS_THROW_IF_NOT(bbs % 32 == 0);
//     M2 = roundup(M, 2);
//     ntotal2 = roundup(ntotal, bbs);

//     codes.resize(ntotal2 * M2 / 2);

//     // printf("M=%d M2=%d code_size=%d\n", M, M2, aq->code_size);
//     pq4_pack_codes(orig.codes.data(), ntotal, M, ntotal2, bbs, M2,
//     codes.get());
// }

void IndexAQFastScan::train(idx_t n, const float* x) {
    if (is_trained) {
        return;
    }
    aq->train(n, x);

    if (metric_type == METRIC_L2) {
        // NOTE: We should quantize the norm of reconstructed vectors
        // rather than the norm of original vectors!!!
        std::vector<float> decoded_x(n * d);
        std::vector<uint8_t> x_codes(n * aq->code_size);
        aq->compute_codes(x, x_codes.data(), n);
        aq->decode(x_codes.data(), decoded_x.data(), n);

        std::vector<float> norms(n, 0);
        fvec_norms_L2sqr(norms.data(), decoded_x.data(), d, n);
        mean_norm = 0;
        for (const auto& norm : norms) {
            mean_norm += norm;
        }
        mean_norm /= n;
        for (auto& norm : norms) {
            norm -= mean_norm;
        }

        norm_aq->train(n, norms.data());
    }

    is_trained = true;
}

void IndexAQFastScan::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);

    AlignedTable<uint8_t> tmp_codes(n * code_size);
    compute_codes(tmp_codes.get(), n, x);

    ntotal2 = roundup(ntotal + n, bbs);
    size_t new_size = ntotal2 * M2 / 2; // assume nbits = 4
    size_t old_size = codes.size();
    if (new_size > old_size) {
        codes.resize(new_size);
        memset(codes.get() + old_size, 0, new_size - old_size);
    }

    pq4_pack_codes_range(
            tmp_codes.get(), M, ntotal, ntotal + n, bbs, M2, codes.get());

    ntotal += n;

    if (implem == 0x22 || implem == 2) {
        orig_codes.resize(n * code_size);
        memcpy(orig_codes.data(),
               tmp_codes.get(),
               sizeof(uint8_t) * orig_codes.size());
    }
}

void IndexAQFastScan::compute_codes(uint8_t* tmp_codes, idx_t n, const float* x)
        const {
    if (n > chunk_size) {
        bool aq_verbose = aq->verbose;
        bool norm_verbose = norm_aq->verbose;
        size_t n_chunks = (n + chunk_size - 1) / chunk_size;
        for (size_t i = 0; i < n_chunks; i++) {
            size_t ni = std::min(chunk_size, n - i * chunk_size);
            if (aq_verbose) {
                printf("\r\tencoding %zd/%zd ...", i * chunk_size + ni, n);
                fflush(stdout);
                if (i == n_chunks - 1 || i == 0) {
                    printf("\n");
                }
            }
            compute_codes(
                    tmp_codes + i * chunk_size * code_size,
                    ni,
                    x + i * chunk_size * d);
            aq->verbose = false;
            norm_aq->verbose = false;
        }
        aq->verbose = aq_verbose;
        norm_aq->verbose = norm_verbose;
        return;
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        aq->compute_codes(x, tmp_codes, n);
    } else {
        // encode vectors
        std::vector<uint8_t> x_codes(n * aq->code_size);
        aq->compute_codes(x, x_codes.data(), n);

        // encode norms
        std::vector<float> norms(n, 0);
        std::vector<float> decoded_x(n * d);

        aq->decode(x_codes.data(), decoded_x.data(), n);
        fvec_norms_L2sqr(norms.data(), decoded_x.data(), d, n);

        for (auto& norm : norms) {
            norm -= mean_norm;
        }

        std::vector<uint8_t> norm_codes(n * norm_aq->code_size);
        norm_aq->compute_codes(norms.data(), norm_codes.data(), n);

        // combine
#pragma omp parallel for if (n > 1000)
        for (idx_t i = 0; i < n; i++) {
            BitstringReader bsr1(
                    x_codes.data() + i * aq->code_size, aq->code_size);
            BitstringReader bsr2(
                    norm_codes.data() + i * norm_aq->code_size,
                    norm_aq->code_size);
            BitstringWriter bsw(tmp_codes + i * code_size, code_size);

            for (size_t m = 0; m < aq->M; m++) {
                int32_t c = bsr1.read(nbits);
                bsw.write(c, nbits);
            }
            for (size_t m = 0; m < norm_aq->M; m++) {
                int32_t c = bsr2.read(nbits);
                bsw.write(c, nbits);
            }
        }
    }
}

void IndexAQFastScan::reset() {
    codes.resize(0);
    ntotal = 0;
}

namespace {

// from impl/ProductQuantizer.cpp
template <class C, typename dis_t>
void aq_estimators_from_tables_generic(
        const IndexAQFastScan& index,
        const uint8_t* codes,
        size_t ncodes,
        const dis_t* dis_table,
        size_t k,
        typename C::T* heap_dis,
        int64_t* heap_ids) {
    using accu_t = typename C::T;

    for (size_t j = 0; j < ncodes; ++j) {
        BitstringReader bsr(codes + j * index.code_size, index.code_size);
        accu_t dis = 0;
        const dis_t* __restrict dt = dis_table;
        for (size_t m = 0; m < index.M; m++) {
            uint64_t c = bsr.read(index.nbits);
            dis += dt[c];
            dt += index.ksub;
        }

        if (C::cmp(heap_dis[0], dis)) {
            heap_pop<C>(k, heap_dis, heap_ids);
            heap_push<C>(k, heap_dis, heap_ids, dis, j);
        }
    }
}

template <class VectorDistance, class ResultHandler>
void search_with_decompress(
        const IndexAQFastScan& index,
        size_t ntotal,
        const float* xq,
        VectorDistance& vd,
        ResultHandler& res) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    const uint8_t* codes = index.orig_codes.data();

#pragma omp parallel for
    for (int64_t q = 0; q < res.nq; q++) {
        SingleResultHandler resi(res);
        resi.begin(q);
        std::vector<float> tmp(index.d);
        const float* x = xq + index.d * q;
        for (size_t i = 0; i < ntotal; i++) {
            index.aq->decode(codes + i * index.code_size, tmp.data(), 1);
            float dis = vd(x, tmp.data());
            resi.add_result(dis, i);
        }
        resi.end();
    }
}

} // anonymous namespace

using namespace quantize_lut;

void IndexAQFastScan::compute_quantized_LUT(
        idx_t n,
        const float* x,
        uint8_t* lut,
        float* normalizers) const {
    size_t dim12 = ksub * M;
    std::unique_ptr<float[]> dis_tables(new float[n * dim12]);
    compute_LUT(dis_tables.get(), n, x);

    for (uint64_t i = 0; i < n; i++) {
        round_uint8_per_column(
                dis_tables.get() + i * dim12,
                M,
                ksub,
                &normalizers[2 * i],
                &normalizers[2 * i + 1]);
    }

    for (uint64_t i = 0; i < n; i++) {
        const float* t_in = dis_tables.get() + i * dim12;
        uint8_t* t_out = lut + i * M2 * ksub;

        for (int j = 0; j < dim12; j++) {
            t_out[j] = int(t_in[j]);
        }
        memset(t_out + dim12, 0, (M2 - M) * ksub);
    }
}

/******************************************************************************
 * Search driver routine
 ******************************************************************************/

void IndexAQFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    if (implem == 0x22) {
        if (metric_type == METRIC_L2) {
            using VD = VectorDistance<METRIC_L2>;
            VD vd = {size_t(d), metric_arg};
            HeapResultHandler<VD::C> rh(n, distances, labels, k);
            search_with_decompress(*this, ntotal, x, vd, rh);
        } else {
            using VD = VectorDistance<METRIC_INNER_PRODUCT>;
            VD vd = {size_t(d), metric_arg};
            HeapResultHandler<VD::C> rh(n, distances, labels, k);
            search_with_decompress(*this, ntotal, x, vd, rh);
        }
        return;
    }

    if (metric_type == METRIC_L2) {
        search_dispatch_implem<true>(n, x, k, distances, labels);
    } else {
        search_dispatch_implem<false>(n, x, k, distances, labels);
    }
}

template <bool is_max>
void IndexAQFastScan::search_dispatch_implem(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    using Cfloat = typename std::conditional<
            is_max,
            CMax<float, int64_t>,
            CMin<float, int64_t>>::type;

    using C = typename std::
            conditional<is_max, CMax<uint16_t, int>, CMin<uint16_t, int>>::type;

    if (n == 0) {
        return;
    }

    // actual implementation used
    int impl = implem;

    if (impl == 0) {
        if (bbs == 32) {
            impl = 12;
        } else {
            impl = 14;
        }
        if (k > 20) {
            impl++;
        }
    }

    if (implem == 1) {
        FAISS_THROW_MSG("Not implemented yet.");
        // FAISS_THROW_IF_NOT(orig_codes);
        // FAISS_THROW_IF_NOT(is_max);
        // float_maxheap_array_t res = {size_t(n), size_t(k), labels,
        // distances}; aq->search(x, n, orig_codes, ntotal, &res, true);
    } else if (implem == 2 || implem == 3 || implem == 4) {
        FAISS_THROW_IF_NOT(!orig_codes.empty());

        const size_t dim12 = ksub * M;
        std::unique_ptr<float[]> dis_tables(new float[n * dim12]);
        compute_LUT(dis_tables.get(), n, x);

        std::vector<float> normalizers(n * 2);

        if (implem == 2) {
            // default float
        } else if (implem == 3 || implem == 4) {
            for (uint64_t i = 0; i < n; i++) {
                round_uint8_per_column(
                        dis_tables.get() + i * dim12,
                        M,
                        ksub,
                        &normalizers[2 * i],
                        &normalizers[2 * i + 1]);
            }
        }

#pragma omp parallel for if (n > 1000)
        for (int64_t i = 0; i < n; i++) {
            int64_t* heap_ids = labels + i * k;
            float* heap_dis = distances + i * k;

            heap_heapify<Cfloat>(k, heap_dis, heap_ids);

            aq_estimators_from_tables_generic<Cfloat>(
                    *this,
                    orig_codes.data(),
                    ntotal,
                    dis_tables.get() + i * dim12,
                    k,
                    heap_dis,
                    heap_ids);

            heap_reorder<Cfloat>(k, heap_dis, heap_ids);

            if (implem == 4) {
                float a = normalizers[2 * i];
                float b = normalizers[2 * i + 1];

                for (int j = 0; j < k; j++) {
                    heap_dis[j] = heap_dis[j] / a + b;
                }
            }
        }
    } else if (impl >= 12 && impl <= 15) {
        FAISS_THROW_IF_NOT(ntotal < INT_MAX);
        int nt = std::min(omp_get_max_threads(), int(n));
        if (nt < 2) {
            if (impl == 12 || impl == 13) {
                search_implem_12<C>(n, x, k, distances, labels, impl);
            } else {
                search_implem_14<C>(n, x, k, distances, labels, impl);
            }
        } else {
            // explicitly slice over threads
#pragma omp parallel for num_threads(nt)
            for (int slice = 0; slice < nt; slice++) {
                idx_t i0 = n * slice / nt;
                idx_t i1 = n * (slice + 1) / nt;
                float* dis_i = distances + i0 * k;
                idx_t* lab_i = labels + i0 * k;
                if (impl == 12 || impl == 13) {
                    search_implem_12<C>(
                            i1 - i0, x + i0 * d, k, dis_i, lab_i, impl);
                } else {
                    search_implem_14<C>(
                            i1 - i0, x + i0 * d, k, dis_i, lab_i, impl);
                }
            }
        }
    } else {
        FAISS_THROW_FMT("invalid implem %d impl=%d", implem, impl);
    }
}

void IndexAQFastScan::compute_LUT(float* lut, idx_t n, const float* x) const {
    if (metric_type == METRIC_INNER_PRODUCT) {
        aq->compute_LUT(n, x, lut, 1.0f);
    } else {
        // compute inner product look-up tables
        const size_t ip_dim12 = aq->M * ksub;
        const size_t norm_dim12 = norm_aq->M * ksub;
        std::vector<float> ip_lut(n * ip_dim12);
        aq->compute_LUT(n, x, ip_lut.data(), -2.0f);

        // norm look-up tables
        const float* norm_lut = norm_aq->codebooks.data();
        FAISS_THROW_IF_NOT(norm_aq->codebooks.size() == norm_dim12);

        // combine them
        for (idx_t i = 0; i < n; i++) {
            memcpy(lut, ip_lut.data() + i * ip_dim12, ip_dim12 * sizeof(*lut));
            lut += ip_dim12;
            memcpy(lut, norm_lut, norm_dim12 * sizeof(*lut));
            lut += norm_dim12;
        }
    }
}

template <class C>
void IndexAQFastScan::search_implem_12(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl) const {
    FAISS_THROW_IF_NOT(bbs == 32);

    // handle qbs2 blocking by recursive call
    int64_t qbs2 = this->qbs == 0 ? 11 : pq4_qbs_to_nq(this->qbs);
    if (n > qbs2) {
        for (int64_t i0 = 0; i0 < n; i0 += qbs2) {
            int64_t i1 = std::min(i0 + qbs2, n);
            search_implem_12<C>(
                    i1 - i0,
                    x + d * i0,
                    k,
                    distances + i0 * k,
                    labels + i0 * k,
                    impl);
        }
        return;
    }

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> quantized_dis_tables(n * dim12);
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    if (skip & 1) {
        quantized_dis_tables.clear();
    } else {
        compute_quantized_LUT(
                n, x, quantized_dis_tables.get(), normalizers.get());
    }

    AlignedTable<uint8_t> LUT(n * dim12);

    // block sizes are encoded in qbs, 4 bits at a time

    // caution: we override an object field
    int qbs = this->qbs;

    if (n != pq4_qbs_to_nq(qbs)) {
        qbs = pq4_preferred_qbs(n);
    }

    int LUT_nq =
            pq4_pack_LUT_qbs(qbs, M2, quantized_dis_tables.get(), LUT.get());
    FAISS_THROW_IF_NOT(LUT_nq == n);

    if (k == 1) {
        SingleResultHandler<C> handler(n, ntotal);
        if (skip & 4) {
            // pass
        } else {
            handler.disable = bool(skip & 2);
            pq4_accumulate_loop_qbs(
                    qbs, ntotal2, M2, codes.get(), LUT.get(), handler);
        }

        handler.to_flat_arrays(distances, labels, normalizers.get());

    } else if (impl == 12) {
        std::vector<uint16_t> tmp_dis(n * k);
        std::vector<int32_t> tmp_ids(n * k);

        if (skip & 4) {
            // skip
        } else {
            HeapHandler<C> handler(
                    n, tmp_dis.data(), tmp_ids.data(), k, ntotal);
            handler.disable = bool(skip & 2);

            pq4_accumulate_loop_qbs(
                    qbs, ntotal2, M2, codes.get(), LUT.get(), handler);

            if (!(skip & 8)) {
                handler.to_flat_arrays(distances, labels, normalizers.get());
            }
        }

    } else { // impl == 13

        ReservoirHandler<C> handler(n, ntotal, k, 2 * k);
        handler.disable = bool(skip & 2);

        if (skip & 4) {
            // skip
        } else {
            pq4_accumulate_loop_qbs(
                    qbs, ntotal2, M2, codes.get(), LUT.get(), handler);
        }

        if (!(skip & 8)) {
            handler.to_flat_arrays(distances, labels, normalizers.get());
        }

        AQFastScan_stats.t0 += handler.times[0];
        AQFastScan_stats.t1 += handler.times[1];
        AQFastScan_stats.t2 += handler.times[2];
        AQFastScan_stats.t3 += handler.times[3];
    }
}

AQFastScanStats AQFastScan_stats;

template <class C>
void IndexAQFastScan::search_implem_14(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl) const {
    FAISS_THROW_IF_NOT(bbs % 32 == 0);

    int qbs2 = qbs == 0 ? 4 : qbs;

    // handle qbs2 blocking by recursive call
    if (n > qbs2) {
        for (int64_t i0 = 0; i0 < n; i0 += qbs2) {
            int64_t i1 = std::min(i0 + qbs2, n);
            search_implem_14<C>(
                    i1 - i0,
                    x + d * i0,
                    k,
                    distances + i0 * k,
                    labels + i0 * k,
                    impl);
        }
        return;
    }

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> quantized_dis_tables(n * dim12);
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    if (skip & 1) {
        quantized_dis_tables.clear();
    } else {
        compute_quantized_LUT(
                n, x, quantized_dis_tables.get(), normalizers.get());
    }

    AlignedTable<uint8_t> LUT(n * dim12);
    pq4_pack_LUT(n, M2, quantized_dis_tables.get(), LUT.get());

    if (k == 1) {
        SingleResultHandler<C> handler(n, ntotal);
        if (skip & 4) {
            // pass
        } else {
            handler.disable = bool(skip & 2);
            pq4_accumulate_loop(
                    n, ntotal2, bbs, M2, codes.get(), LUT.get(), handler);
        }
        handler.to_flat_arrays(distances, labels, normalizers.get());

    } else if (impl == 14) {
        std::vector<uint16_t> tmp_dis(n * k);
        std::vector<int32_t> tmp_ids(n * k);

        if (skip & 4) {
            // skip
        } else if (k > 1) {
            HeapHandler<C> handler(
                    n, tmp_dis.data(), tmp_ids.data(), k, ntotal);
            handler.disable = bool(skip & 2);

            pq4_accumulate_loop(
                    n, ntotal2, bbs, M2, codes.get(), LUT.get(), handler);

            if (!(skip & 8)) {
                handler.to_flat_arrays(distances, labels, normalizers.get());
            }
        }

    } else { // impl == 15

        ReservoirHandler<C> handler(n, ntotal, k, 2 * k);
        handler.disable = bool(skip & 2);

        if (skip & 4) {
            // skip
        } else {
            pq4_accumulate_loop(
                    n, ntotal2, bbs, M2, codes.get(), LUT.get(), handler);
        }

        if (!(skip & 8)) {
            handler.to_flat_arrays(distances, labels, normalizers.get());
        }
    }
}

} // namespace faiss
