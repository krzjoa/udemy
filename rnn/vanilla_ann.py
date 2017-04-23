# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from utils import init_weights, all_parity
from sklearn.utils import shuffle


class DenseLayer(object):

    def __init__(self, M1, M2, layer_id):
        self.layer_id = layer_id
        self.M1 = M1
        self.M2 = M2

        # Inicjalizacja wartości
        W = init_weights(M1, M2)
        b = np.zeros(M2)

        # Wagi i biasy
        self.W = theano.shared(W, "W_%s" % self.layer_id)
        self.b = theano.shared(b, "W_%s" % self.layer_id)

        self.params = [self.W, self.b]


    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)




class VanillaANN(object):

    def __init__(self):
        self.hidden_layers = []


# class VanillaANN(object):
#
#     def __init__(self):
#         self.id = an_id
#
#
#
#     def fit(self):