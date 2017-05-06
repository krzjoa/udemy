# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

x  = 2*np.random.rand(300) + np.sin(np.linspace(0, 3*np.pi, 300))
plt.plot(x)
plt.title('original')
plt.show()

decay = T.scalar('decay')
sequence = T.vector('sequence')

def recurrence(x, last, decay):
    return (1-decay)*x + decay*last

outputs, _ = theano.scan(
    fn=recurrence,
    sequences=sequence,
    n_steps=sequence.shape[0],
    outputs_info=[np.float64(0)],
    non_sequences=[decay]
)

# W Theano function jako outputs zawsze dajemy funkcję

lpf = theano.function(
    inputs=[sequence, decay],
    outputs=outputs
)

Y = lpf(x, 0.99)

print Y

plt.plot(Y)
plt.title("filtered")
plt.show()
