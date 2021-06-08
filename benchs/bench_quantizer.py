# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import faiss
import time
import numpy as np

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M


def eval_codec(q, xb):
    t0 = time.time()
    codes = q.compute_codes(xb)
    t1 = time.time()
    decoded = q.decode(codes)
    return ((xb - decoded) ** 2).sum() / xb.shape[0], t1 - t0


def eval_quantizer(q, xb, xt, name):
    t0 = time.time()
    q.train(xt)
    t1 = time.time()
    train_t = t1 - t0
    err, encode_t = eval_codec(q, xb)
    print(f'===== {name}:')
    print(f'\tmean square error = {err}')
    print(f'\ttraining time: {train_t} s')
    print(f'\tencoding time: {encode_t} s')


todo = sys.argv[1:]
ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()
xt = ds.get_train()

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

M = 8
nbits = 8

if 'lsq' in todo:
    lsq = faiss.LocalSearchQuantizer(d, M, nbits)
    lsq.verbose = True  # show detailed training progress
    eval_quantizer(lsq, xb, xt, 'lsq')

if 'pq' in todo:
    pq = faiss.ProductQuantizer(d, M, nbits)
    eval_quantizer(pq, xb, xt, 'pq')

if 'rq' in todo:
    rq = faiss.ResidualQuantizer(d, M, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.verbose = True
    eval_quantizer(rq, xb, xt, 'rq')

if 'lsq-scalar' in todo:
    lsq = faiss.LocalSearchQuantizer(1, 2, 4)
    lsq.verbose = True  # show detailed training progress
    lsq.lambd = 0.01
    lsq.encode_type = 1
    lsq.train_iters = 25
    # lsq.nperts = 2
    xb_norm = np.sum(xb ** 2, axis=1, keepdims=True)
    xt_norm = np.sum(xt ** 2, axis=1, keepdims=True)
    mean = np.mean(xt_norm)
    print(mean)
    eval_quantizer(lsq, xb_norm - mean, xt_norm - mean, 'lsq')

if 'pq-scalar' in todo:
    pq = faiss.ProductQuantizer(1, 1, 8)
    xb_norm = np.sum(xb ** 2, axis=1, keepdims=True)
    xt_norm = np.sum(xt ** 2, axis=1, keepdims=True)
    mean = np.mean(xt_norm)
    eval_quantizer(pq, xb_norm - mean, xt_norm - mean, 'pq')

if 'rq-scalar' in todo:
    rq = faiss.ResidualQuantizer(1, 2, 4)
    xb_norm = np.sum(xb ** 2, axis=1, keepdims=True)
    xt_norm = np.sum(xt ** 2, axis=1, keepdims=True)
    mean = np.mean(xt_norm)
    eval_quantizer(rq, xb_norm - mean, xt_norm - mean, 'rq')