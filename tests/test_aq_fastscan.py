# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import time

import numpy as np
import faiss

from faiss.contrib import datasets
import platform


class TestCompileOptions(unittest.TestCase):

    def test_compile_options(self):
        options = faiss.get_compile_options()
        options = options.split(' ')
        for option in options:
            assert option in ['AVX2', 'NEON', 'GENERIC', 'OPTIMIZE']


class TestSearch(unittest.TestCase):

    def test_PQ4_accuracy(self):
        d = 32
        M, nbits = 16, 4
        ds  = datasets.SyntheticDataset(d, 2000, 5000, 1000)

        index_gt = faiss.IndexFlatIP(d)
        index_gt.add(ds.get_database())
        Dref, Iref = index_gt.search(ds.get_queries(), 10)

        lsq = faiss.LocalSearchQuantizer(d, M, nbits)
        lsq.lambd = 0.01
        index = faiss.IndexAQFastScan(lsq, faiss.METRIC_INNER_PRODUCT)
        index.implem = 4
        index.aq.verbose = True
        index.train(ds.get_train())
        index.add(ds.get_database())
        Da, Ia = index.search(ds.get_queries(), 10)

        nq = Iref.shape[0]
        recall_a = (Iref[:, 0] == Ia[:, 0]).sum() / nq
        print(f'recall@1 = {recall_a:.3f}')

        index2 = faiss.IndexPQFastScan(d, M, nbits, faiss.METRIC_INNER_PRODUCT)
        index2.implem = 12
        index2.train(ds.get_train())
        index2.add(ds.get_database())
        Db, Ib = index2.search(ds.get_queries(), 10)

        nq = Iref.shape[0]
        recall_b = (Iref[:, 0] == Ib[:, 0]).sum() / nq
        print(f'recall@1 = {recall_b:.3f}')

        assert recall_a > recall_b