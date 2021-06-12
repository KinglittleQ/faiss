# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import time
import numpy as np
import faiss

from faiss.contrib.datasets import DatasetSIFT1M, DatasetGlove


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
ds = DatasetGlove()

xq = ds.get_queries()[:1000]
xb = ds.get_database()
gt = ds.get_groundtruth()[:1000]
# xt = ds.get_train()
xt = xb[:100000]

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

M = 16
nbits = 4


if 'lsq' in todo:
    lsq = faiss.LocalSearchQuantizer(d, M, nbits)
    lsq.lambd = 0.1
    lsq.verbose = True
    index = faiss.IndexAQFastScan(lsq, faiss.METRIC_INNER_PRODUCT)
    index.implem = 12
    index.train(xt)
    index.add(xb)
    evaluate(index)

if 'pq' in todo:
    index = faiss.IndexPQFastScan(d, M, nbits, faiss.METRIC_INNER_PRODUCT)
    index.implem = 12
    index.train(xt)
    index.add(xb)
    eval_quantizer(index.pq, xb, xt, "pq")
    evaluate(index)

if 'rq' in todo:
    rq = faiss.ResidualQuantizer(d, M, nbits)
    rq.verbose = True
    index = faiss.IndexAQFastScan(rq, faiss.METRIC_INNER_PRODUCT)
    index.implem = 12
    index.train(xt)
    index.add(xb)
    evaluate(index)

