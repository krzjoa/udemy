# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from utils import init_weights, all_parity


class SimpleRNN(object):

    def __init__(self, M, learning_rate=10e-1,
                 mu=0.99, reg=1.0, activation=T.tanh,
                 epochs=100, show_fig=False):
        self.M = M
        self.learning_rate = learning_rate
        self.mu = mu
        self.reg = reg
        self.activation = activation
        self.epochs = epochs
        self.show_fig = show_fig


    def fit(self, X, Y):

        # Wielkość pojedynczego element, np. wielkość słownika w one-hot encoding
        D = X[0].shape[1]
        K = len(set(Y.flatten()))
        N = len(Y)

        # Inicjalizacja wag
        Wx = init_weights(D, self.M)
        Wh = init_weights(self.M, self.M)
        bh = np.zeros(self.M)
        h0 = np.zeros(self.M)
        # Tutaj coś nie pasuje. To wygląda tak, jak z pojedynczego elementu sekwencji
        # otrztmywalibyśmy predykcje dla całej sekwencji. Łot da fak?
        Wo = init_weights(self.M, K)
        bo = np.zeros(K)

        # Theano variables
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)

        self.params = [self.Wx, self.Wh, self.bh,
                       self.h0, self.Wo, self.bo]

        thX = T.fmatrix('X')
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            # Zwraca h(t), y(t)
            h_t = self.activation(x_t.dot(self.Wx) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y] = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=thX,
            n_steps=thX.shape[0]
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0], thY)]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + self.mu*dp - self.learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, self.mu*dp - self.learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction, y],
            updates=updates
        )

        costs = []

        for i in xrange(self.epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0

            for j in xrange(N):
                c, p, rout = self.train_op(X[j], Y[j])
                cost += c
                if p[-1] == Y[j, -1]:
                    n_correct += 1
            print "Shape y:", rout.shape
            print "i", i, "cost:", cost, "classification rate:", (float(n_correct))
            costs.append(cost)

        if self.show_fig:
            plt.plot(costs)
            plt.show()




def parity(B=12, learning_rate=10e-5, epochs=200):
    X, Y = all_parity(B)
    N, t = X.shape

    Y_t = np.zeros(X.shape, dtype=np.int32)
    for n in xrange(N):
        ones_count = 0
        for i in xrange(t):
            if X[n, i] == 1:
                ones_count += 1
            if ones_count % 2 == 1:
                Y_t[n, i] = 1

    X = X.reshape(N, t, 1).astype(np.float32)

    rnn = SimpleRNN(4, learning_rate=learning_rate, epochs=epochs,
                    activation=T.nnet.sigmoid)
    rnn.fit(X, Y)


if __name__ == "__main__":
    parity()




