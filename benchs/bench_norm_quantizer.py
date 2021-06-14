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
    t0 = time.time()
    q.train(xt)
    t1 = time.time()
    train_t = t1 - t0

    err, encode_t = eval_codec(q, xb)

    print(f'===== {name}:')
    print(f'\tmean square error = {err}')
    print(f'\ttraining time: {train_t} s')
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

verbose = bool(int(sys.argv[1]))
todo = sys.argv[2:]
if len(todo) == 0:
    todo = ['lsq-bf', 'lsq-icm', 'rq', 'pq']

ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()
xt = ds.get_train()

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

M, nbits = 14, 4
if 'norm' in todo:
    q = faiss.LocalSearchQuantizer(d, M, nbits)
    q.lambd = 0.1
    q.verbose = True
    q.train(xt)
    codes = q.compute_codes(xt)
    decoded_xt = q.decode(codes)
    np.save(f'decoded_xt_{M}_{nbits}.npy', decoded_xt)
else:
    decoded_xt = np.load(f'decoded_xt_{M}_{nbits}.npy')

xt_norm = np.sum(decoded_xt ** 2, axis=1, keepdims=True)  # [nb, 1]
mean_norm = xt_norm.mean()

xt_norm -= mean_norm

print(mean_norm)
print(xt_norm[:10])

M = 2
nbits = 4

if 'lsq-bf' in todo:
    aq = faiss.LocalSearchQuantizer(1, M, nbits)
    aq.lambd = 0.01
    aq.train_iters = 10
    aq.encode_type = 1
    aq.p = 0.1
    aq.verbose = verbose
    eval_quantizer(aq, xt_norm, xt_norm, "lsq-brutefoce")

if 'lsq-icm' in todo:
    aq = faiss.LocalSearchQuantizer(1, M, nbits)
    aq.lambd = 0.01
    aq.train_iters = 10
    aq.encode_type = 0
    aq.p = 0.1
    aq.verbose = verbose
    eval_quantizer(aq, xt_norm, xt_norm, "lsq-icm")

if 'rq' in todo:
    aq = faiss.ResidualQuantizer(1, M, nbits)
    aq.verbose = verbose
    eval_quantizer(aq, xt_norm, xt_norm, "rq")

if 'pq-8' in todo:
    pq = faiss.ProductQuantizer(1, 1, M * nbits)
    pq.cp.niter = 200
    pq.verbose = verbose
    eval_quantizer(pq, xt_norm, xt_norm, "pq")