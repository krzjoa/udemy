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
        W = init_weights(M1, M2).astype(np.float32)
        b = np.zeros(M2).astype(np.float32)

        # Wagi i biasy
        self.W = theano.shared(W, "W_%s" % self.layer_id)
        self.b = theano.shared(b, "W_%s" % self.layer_id)

        self.params = [self.W, self.b]


    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)


class VanillaANN(object):

    def __init__(self, layers, learning_rate=10e-3, mu=0.99,
                 reg=10e-12, eps=10e-10, epochs=10, batch_size=32,
                 print_period=10, show_fig=True):

        self.hidden_layers_sizes = layers
        self.epochs = epochs

        # Params
        self.reg = np.float32(reg)
        self.eps = np.float32(eps)
        self.learning_rate = np.float32(learning_rate)
        self.mu = np.float32(mu)
        self.batch_size = batch_size
        self.print_period=print_period
        self.show_fig = show_fig

    def fit(self, X, Y):
        Y = Y.astype(np.int32)

        N, D = X.shape
        K = len(set(Y))

        self.hidden_layers = []

        # Rozmiar wyjścia z warstwy poprzedniej jest rozmiarem wejścia następnej
        M1 = D
        count = 0

        for M2 in self.hidden_layers_sizes:
            h = DenseLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        # Ostatnia wartswa x liczba klasd
        W = init_weights(M1, K).astype(np.float32)
        b = np.zeros(K).astype(np.float32)

        # Theano
        self.W =theano.shared(W, "W_logreg")
        self.b = theano.shared(b, "b_logreg")

        self.params = [self.W, self.b]
        for layer in self.hidden_layers:
            self.params += layer.params

        dparams = [theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in self.params]

        # Zmienne Theano
        thX = T.matrix('X', dtype='float32')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        # Funkcja kosztu
        rcost = self.reg*T.sum([(p**2).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
#        cost = -T.mean(T.log(pY[T.arange(thY.shape[0], thY)])) + rcost
        prediction = self.predict(thX)
        grads = T.grad(cost, self.params)

        updates = [
            (p, p + self.mu*dp - self.learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, self.mu*dp - self.learning_rate*g)  for dp, g in zip(dparams, grads)
        ]

        train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates
        )

        n_batches = N / self.batch_size
        costs = []

        for i in xrange(self.epochs):
            X, Y = shuffle(X, Y)
            for j in xrange(n_batches):
                Xbatch = X[j*self.batch_size: (j*self.batch_size + self.batch_size)]
                Ybatch = Y[j*self.batch_size: (j*self.batch_size + self.batch_size)]

                c, p = train_op(Xbatch, Ybatch)

                print "Cost", c

                # if j % self.print_period == 0:
                #     costs.append(c)
                #     e = np.mean(Ybatch != p)
                #     print "i:", i,

            # if self.show_fig:
            #     plt.plot(costs)
            #     plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return T.argmax(pY, axis=1)

def wide():
    X, Y = all_parity(12)
    model = VanillaANN([2048])
    model.fit(X, Y)

def depp():
    X, Y = all_parity(12)
    model = VanillaANN([1024]*2)
    model.fit(X, Y)

if __name__ == '__main__':
    #wide()
    depp()

    # Coś tu nie pasi...