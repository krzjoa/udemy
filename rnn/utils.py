# -*- coding: utf-8 -*-
import numpy as np


def init_weights(Mi, Mo):
    return np.random.rand(Mi, Mo) / np.sqrt(Mi + Mo)

def all_parity(nbits=4):
    # Liczba przykładów
    N = 2 ** nbits
    X = np.zeros((N, nbits))
    Y = np.zeros(N)

    for row in xrange(N):
        i = row / N
        # Bity
        for j in xrange(nbits):
            if i % (2**(i+1)) != 0:
                i -= 2**j
                X[row, j] = 1
        Y[row] = X[row].sum() % 2
    return X, Y