from asgn2.layers import *
from asgn2.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache

def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db

def affine_leakyrelu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, leakyrelu_cache = leakyrelu_forward(a)
    cache = (fc_cache, leakyrelu_cache)
    return out, cache

def affine_leakyrelu_backward(dout, cache):

    fc_cache, leakyrelu_cache = cache
    da = leakyrelu_backward(dout, leakyrelu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_softplus_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, softplus_cache = softplus_forward(a)
    cache = (fc_cache, softplus_cache)
    return out, cache


def affine_softplus_backward(dout, cache):

    fc_cache, softplus_cache = cache
    da = softplus_backward(dout, softplus_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_elu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, elu_cache = elu_forward(a)
    cache = (fc_cache, elu_cache)
    return out, cache

def affine_elu_backward(dout, cache):

    fc_cache, elu_cache = cache
    da = elu_backward(dout, elu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_selu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, selu_cache = selu_forward(a)
    cache = (fc_cache, selu_cache)
    return out, cache

def affine_selu_backward(dout, cache):

    fc_cache, selu_cache = cache
    da = selu_backward(dout, selu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

