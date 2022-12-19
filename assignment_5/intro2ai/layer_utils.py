from .layers import *


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    out1, cache1 = conv_forward_naive(x, w, b, conv_param)
    out2, cache2 = relu_forward(out1)
    out, cache3 = max_pool_forward_naive(out2, pool_param)
    cache = (cache1,cache2,cache3)
    return out, cache

def conv_relu_pool_backward(dout, cache):
    dx = max_pool_backward_naive(dout, cache[2])
    dx = relu_backward(dx, cache[1])
    dx, dw, db = conv_backward_naive(dx, cache[0])
    return dx, dw, db

def conv_relu_forward(x, w, b, conv_param):
    out1, cache1 = conv_forward_naive(x, w, b, conv_param)
    out, cache2 = relu_forward(out1)
    cache = (cache1,cache2)
    return out, cache

def conv_relu_backward(dout, cache):
    dx = relu_backward(dout, cache[1])
    dx, dw, db = conv_backward_naive(dx, cache[0])
    return dx, dw, db

