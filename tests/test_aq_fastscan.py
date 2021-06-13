# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import time

import numpy as np
import faiss

from faiss.contrib import datasets


sp = faiss.swig_ptr



def compute_LUT_IP_ref(xq, codebooks, alpha=1.0):
    M, ksub, d = codebooks.shape
    codebooks = codebooks.reshape((-1, d))
    lut = alpha * xq.dot(codebooks.T)  # [n, M * ksub]
    lut = lut.reshape((-1, M, ksub))
    return lut


def compute_LUT_L2_ref(xq, codebooks, norm_codebooks):
    M1, ksub, d = codebooks.shape
    M2 = norm_codebooks.shape[0]
    nq = xq.shape[0]

    lut0 = compute_LUT_IP_ref(xq, codebooks, -2.0)  # [n, M1, ksub]
    lut1 = np.repeat(norm_codebooks[np.newaxis, :, :], nq, axis=0)  # [n, M2, ksub]
    lut = np.concatenate((lut0, lut1), axis=1)
    return lut


class TestComponents(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        d = 32
        M, nbits = 16, 4
        ds = datasets.SyntheticDataset(d, 2000, 5000, 1000)
        self.d = d
        self.M = M
        self.nbits = nbits
        self.ds = ds
        self.xt = ds.get_train()
        self.xb = ds.get_database()
        self.xq = ds.get_queries()

    def test_compute_lut_IP(self):
        d, M, nbits = self.d, self.M, self.nbits
        ksub = (1 << nbits)
        xt, xb, xq = self.xt, self.xb, self.xq
        nt, nb, nq = xt.shape[0], xb.shape[0], xq.shape[0]

        aq = faiss.LocalSearchQuantizer(d, M, nbits)
        aq.lambd = 0.1
        index = faiss.IndexAQFastScan(aq, faiss.METRIC_INNER_PRODUCT)
        index.train(xt)

        lut = np.zeros((nq, M, ksub), dtype=np.float32)
        index.compute_LUT(sp(lut), nq, sp(xq))

        # ref
        codebooks = faiss.vector_to_array(aq.codebooks)
        codebooks = codebooks.reshape((M, ksub, d))
        lut_ref = compute_LUT_IP_ref(xq, codebooks)

        np.testing.assert_allclose(lut, lut_ref, rtol=1e-6)

    def test_compute_lut_L2(self):
        d, M, nbits = self.d, self.M, self.nbits
        ksub = (1 << nbits)
        xt, xb, xq = self.xt, self.xb, self.xq
        nt, nb, nq = xt.shape[0], xb.shape[0], xq.shape[0]

        aq = faiss.LocalSearchQuantizer(d, M - 2, nbits)
        aq.lambd = 0.1
        norm_aq = faiss.LocalSearchQuantizer(1, 2, nbits)
        norm_aq.lambd = 0.1
        norm_aq.encode_type = 1
        index = faiss.IndexAQFastScan(aq, faiss.METRIC_L2, 32, norm_aq)
        index.train(xt)

        lut = np.zeros((nq, M, ksub), dtype=np.float32)
        index.compute_LUT(sp(lut), nq, sp(xq))

        # ref
        codebooks = faiss.vector_to_array(aq.codebooks)
        codebooks = codebooks.reshape((M - 2, ksub, d))
        norm_codebooks = faiss.vector_to_array(norm_aq.codebooks)
        norm_codebooks = norm_codebooks.reshape((2, ksub))
        lut_ref = compute_LUT_L2_ref(xq, codebooks, norm_codebooks)

        np.testing.assert_allclose(lut, lut_ref, rtol=1e-6)
