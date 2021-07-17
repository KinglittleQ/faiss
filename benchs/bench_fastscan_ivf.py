# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import time
import numpy as np
import faiss

from faiss.contrib.datasets import DatasetSIFT1M


def eval_codec(q, xb):
    t0 = time.time()
    codes = q.compute_codes(xb)
    t1 = time.time()
    decoded = q.decode(codes)
    return ((xb - decoded) ** 2).sum() / xb.shape[0], t1 - t0


def eval_quantizer(q, xb, xt, name):
    err, encode_t = eval_codec(q, xb)
    print(f'===== {name}:')
    print(f'\tmean square error = {err}')
    print(f'\tencoding time: {encode_t} s')


def evaluate(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    recall = 1. * faiss.eval_intersection(I, gt[:, :k]) / (nq * k)
    print("\t%.5f ms per query,\tR@%d %.4f" % (
        (t1 - t0) * 1000.0 / nq, k, recall))


def evaluate_ivf(index):
    index.train(xt)
    index.add(xb)
    for nprobe in 1, 2, 4, 8, 16, 32, 64, 128, 256:
        print(f"\t nprobe {nprobe},", end='')
        index.nprobe = nprobe
        evaluate(index)


def evaluate_ivf_refine(index):
    index.train(xt)
    index.add(xb)
    for k_factor in 1, 2, 4, 8, 16, 32, 64, 128:
        for nprobe in 1, 2, 4, 8, 16, 32, 64, 128, 256:
            print(f"\t k_factor {k_factor},\tnprobe {nprobe},", end='')
            base_index = faiss.downcast_index(index.base_index)
            base_index.nprobe = nprobe
            index.k_factor = k_factor
            evaluate(index)


def create_aq(name, d, M, nbits):
    if name == 'lsq':
        aq = LSQ(d, M, nbits)
        aq.icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
        if M <= 3:
            aq.encode_type = LSQ.EncodeType_BF
        else:
            aq.encode_ils_iters = 64
    elif name == 'rq':
        aq = RQ(d, M, nbits)

    return aq


def create_aq_ivf(name1, M1, name2, M2):
    aq = create_aq(name1, d, M1, nbits)
    norm_aq = create_aq(name2, 1, M2, nbits)

    index0 = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFAQFastScan(index0, aq, norm_aq, d, nlist, faiss.METRIC_L2)

    # do not release the memory
    index.alloc_aq = aq
    index.alloc_norm_aq = norm_aq
    index.alloc_index0 = index0

    return index


def is_aq_ivf(s):
    ss = s.split('-')
    if len(ss) == 4:
        return (ss[0] in ['rq', 'lsq']) and (ss[2] in ['rq', 'lsq'])
    return False


def is_aq_ivf_refine(s):
    ss = s.split('-')
    if len(ss) == 5:
        return (ss[0] in ['rq', 'lsq']) and (ss[2] in ['rq', 'lsq']) and (ss[4] == 'refine')
    return False



k = int(sys.argv[1])
todo = sys.argv[2:]
ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth().astype(np.int64)
xt = ds.get_train()

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

M = 16  # 64 bits

nbits = 4
nlist = 1024

ngpus = faiss.get_num_gpus()

LSQ = faiss.LocalSearchQuantizer
RQ = faiss.ResidualQuantizer


# lsq-lsq, lsq-rq, rq-lsq, rq-lsq, **-**-refine
for s in todo:
    if is_aq_ivf(s):
        print(f'========== {s}')
        ss = s.split('-')
        index = create_aq_ivf(ss[0], int(ss[1]), ss[2], int(ss[3]))
        index.verbose = False
        evaluate_ivf(index)
    elif is_aq_ivf_refine(s):
        print(f'========== {s}')
        ss = s.split('-')
        base_index = create_aq_ivf(ss[0], int(ss[1]), ss[2], int(ss[3]))
        index = faiss.IndexRefineFlat(base_index)
        evaluate_ivf_refine(index)


if 'hnsw' in todo:
    print('========== hnsw')
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200
    index.add(xb)

    for efSearch in 2, 4, 8, 16, 32, 64, 128, 256:
        print(f"\t efSearch {efSearch},", end='')
        index.hnsw.efSearch = efSearch
        evaluate(index)


if 'pq' in todo:
    print('========== pq')
    coarse_index = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFPQFastScan(coarse_index, d, nlist, M, nbits, faiss.METRIC_L2)
    evaluate_ivf(index)


if 'pq-8' in todo:
    print('========== pq-8')
    coarse_index = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFPQ(coarse_index, d, nlist, M // 2, nbits * 2, faiss.METRIC_L2)
    evaluate_ivf(index)
