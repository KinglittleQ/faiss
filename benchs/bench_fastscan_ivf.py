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


def evaluate_nprobe(index, nprobe):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)

    t0 = time.time()
    index.nprobe = nprobe
    D, I = index.search(xq, k)
    t1 = time.time()

    recall = 1. * faiss.eval_intersection(I, gt[:, :k]) / (nq * k)
    print("\t nprobe %d,\t%.5f ms per query,\tR@%d %.4f" % (
        nprobe, (t1 - t0) * 1000.0 / nq, k, recall))


def evaluate(index):
    index.train(xt)
    index.add(xb)
    for nprobe in 1, 2, 4, 8, 16, 32, 64, 128, 256:
        evaluate_nprobe(index, nprobe)



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

vec_M = 14
norm_M = 2
M = vec_M + norm_M

nbits = 4
nlist = 1024

ngpus = faiss.get_num_gpus()

LSQ = faiss.LocalSearchQuantizer
RQ = faiss.ResidualQuantizer


if 'lsq-lsq' in todo:
    lsq = LSQ(d, vec_M, nbits)
    lsq.icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
    norm_aq = LSQ(1, norm_M, nbits)
    norm_aq.encode_type = LSQ.EncodeType_BF

    coarse_index = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFAQFastScan(coarse_index, lsq, norm_aq, d, nlist, faiss.METRIC_L2)
    index.implem = 12
    index.by_residual = True
    evaluate(index)

if 'lsq-rq' in todo:
    lsq = LSQ(d, vec_M, nbits)
    lsq.icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
    norm_aq = RQ(1, norm_M, nbits)

    coarse_index = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFAQFastScan(coarse_index, lsq, norm_aq, d, nlist, faiss.METRIC_L2)
    index.implem = 12
    evaluate(index)

if 'pq' in todo:
    coarse_index = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFPQFastScan(coarse_index, d, nlist, M, nbits, faiss.METRIC_L2)

    index.implem = 12
    evaluate(index)

if 'rq-rq' in todo:
    rq = RQ(d, vec_M, nbits)
    norm_aq = RQ(1, norm_M, nbits)

    coarse_index = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFAQFastScan(coarse_index, rq, norm_aq, d, nlist, faiss.METRIC_L2)
    index.implem = 12
    evaluate(index)

if 'rq-lsq' in todo:
    rq = RQ(d, vec_M, nbits)
    norm_aq = LSQ(1, norm_M, nbits)
    norm_aq.encode_type = LSQ.EncodeType_BF

    coarse_index = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFAQFastScan(coarse_index, rq, norm_aq, d, nlist, faiss.METRIC_L2)
    index.implem = 12
    evaluate(index)

if 'pq-8' in todo:
    coarse_index = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFPQ(coarse_index, d, nlist, M // 2, nbits * 2, faiss.METRIC_L2)
    index.by_residual = True

    evaluate(index)
