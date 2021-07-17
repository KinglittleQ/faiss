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
    print("\t %7.3f ms per query, R@%d %.4f" % (
        (t1 - t0) * 1000.0 / nq, k, recall))


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


def create_aq_fastscan(name1, M1, name2, M2):
    aq = create_aq(name1, d, M1, nbits)
    norm_aq = create_aq(name2, 1, M2, nbits)
    index = faiss.IndexAQFastScan(aq, norm_aq, faiss.METRIC_L2)

    # do not release the memory
    index.alloc_aq = aq
    index.alloc_norm_aq = norm_aq

    return index


def is_aq_fastscan(s):
    ss = s.split('-')
    if len(ss) == 4:
        return (ss[0] in ['rq', 'lsq']) and (ss[2] in ['rq', 'lsq'])
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

M = 16
nbits = 4

LSQ = faiss.LocalSearchQuantizer
RQ = faiss.ResidualQuantizer
ngpus = faiss.get_num_gpus()


# lsq-lsq, lsq-rq, rq-lsq, rq-lsq
for s in todo:
    if is_aq_fastscan(s):
        print(f'========== {s}')
        ss = s.split('-')
        index = create_aq_fastscan(ss[0], int(ss[1]), ss[2], int(ss[3]))
        index.verbose = True
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


if 'pq-8' in todo:
    index = faiss.IndexPQ(d, M // 2, nbits * 2, faiss.METRIC_L2)
    index.train(xt)
    index.add(xb)

    eval_quantizer(index.pq, xb, xt, "pq-8")
    evaluate(index)
