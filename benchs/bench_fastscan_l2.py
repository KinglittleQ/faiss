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

    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))


k = int(sys.argv[1])
todo = sys.argv[2:]
ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()
xt = ds.get_train()

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

vec_M = 14
norm_M = 2
M = vec_M + norm_M

nbits = 4


if 'lsq-lsq' in todo:
    lsq = faiss.LocalSearchQuantizer(d, vec_M, nbits)
    lsq.lambd = 0.1
    lsq.verbose = True

    norm_aq = faiss.LocalSearchQuantizer(1, norm_M, nbits)
    norm_aq.lambd = 0.01
    norm_aq.train_iters = 10
    norm_aq.encode_type = 1
    norm_aq.p = 0.1
    norm_aq.verbose = True

    index = faiss.IndexAQFastScan(lsq, norm_aq, faiss.METRIC_L2)
    # index.implem = 0x22
    index.implem = 12
    index.train(xt)
    index.add(xb)
    evaluate(index)

if 'lsq-rq' in todo:
    lsq = faiss.LocalSearchQuantizer(d, vec_M, nbits)
    lsq.lambd = 0.1
    lsq.verbose = True

    norm_aq = faiss.ResidualQuantizer(1, norm_M, nbits)
    norm_aq.verbose = True

    index = faiss.IndexAQFastScan(lsq, norm_aq, faiss.METRIC_L2)
    # index.implem = 0x22
    index.implem = 12
    index.train(xt)
    index.add(xb)
    evaluate(index)

if 'pq' in todo:
    index0 = faiss.IndexPQ(d, M, nbits, faiss.METRIC_L2)
    index0.train(xt)
    index0.add(xb)

    index = faiss.IndexPQFastScan(index0)
    index.implem = 12
    eval_quantizer(index.pq, xb, xt, "pq")
    evaluate(index)

if 'rq-rq' in todo:
    rq = faiss.ResidualQuantizer(d, vec_M, nbits)
    rq.verbose = True

    norm_aq = faiss.ResidualQuantizer(1, norm_M, nbits)
    norm_aq.verbose = True

    index = faiss.IndexAQFastScan(rq, norm_aq, faiss.METRIC_L2)
    index.implem = 12
    index.train(xt)
    index.add(xb)
    evaluate(index)

if 'rq-lsq' in todo:
    rq = faiss.ResidualQuantizer(d, vec_M, nbits)
    rq.verbose = True

    norm_aq = faiss.LocalSearchQuantizer(1, norm_M, nbits)
    norm_aq.lambd = 0.01
    norm_aq.train_iters = 10
    norm_aq.encode_type = 1
    norm_aq.p = 0.1
    norm_aq.verbose = True

    index = faiss.IndexAQFastScan(rq, norm_aq, faiss.METRIC_L2)
    index.implem = 12
    index.train(xt)
    index.add(xb)
    evaluate(index)

if 'index-rq' in todo:
    index = faiss.IndexResidual(d, vec_M, nbits)
    index.verbose = True
    index.train(xt)
    index.add(xb)
    evaluate(index)

if 'pq-8' in todo:
    index = faiss.IndexPQ(d, M // 2, nbits * 2, faiss.METRIC_L2)
    index.train(xt)
    index.add(xb)

    eval_quantizer(index.pq, xb, xt, "pq-8")
    evaluate(index)
