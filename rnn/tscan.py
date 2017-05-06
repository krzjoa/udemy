import numpy as np
import theano
import theano.tensor as T

x = T.vector('x')

def square(x):
    return x**2

outputs, updates = theano.scan(
    fn=square,
    sequences=x,
    n_steps=x.shape[0]
)


square_op = theano.function(
    inputs=[x],
    outputs=[outputs]
)


print square_op(np.array([1,2,3,4,5,6,7]))