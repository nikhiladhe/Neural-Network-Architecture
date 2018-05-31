import numpy as np

from asgn2.layers import *
from asgn2.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.

  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.D = input_dim
    self.H = hidden_dim
    self.C = num_classes
    
    w1 = weight_scale * np.random.randn(self.D, self.H)    # initialize 1st wt matrix input dimns*hidden dimns
    b1 = np.zeros(hidden_dim)                              # initialize 1st bias matrix to 0
    w2 = weight_scale * np.random.randn(self.H, self.C)    # initialize 2nd wt matrix hidden dimns* no of classes
    b2 = np.zeros(self.C)                                  # initialize 2nd bias matrix to 0

    self.params.update({'W1': w1,'W2': w2,'b1': b1,'b2': b2})
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']

    X = X.reshape(X.shape[0], self.D)                       # X=(no of examples,32*32*3)
    
    
    #HL, cache_HL = affine_relu_forward(X, W1, b1)           # 1st forward pass
    # 999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
    HL, cache_HL = affine_leakyrelu_forward(X, W1, b1)           # 1st forward pass
    
    
    scores, cache_scores = affine_forward(HL, W2, b2)       # 2nd forward pass
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    # calculating loss = loss function loss + regularization loss
    func_loss, dscores = softmax_loss(scores, y)      # here loss function is softmax loss
    reg_loss = 0.5 * self.reg * np.sum(W1**2)         # reg penalty for 1st layer
    reg_loss += 0.5 * self.reg * np.sum(W2**2)        # reg penalty for 2nd layer
    loss = func_loss + reg_loss
    
    dx1, dW2, db2 = affine_backward(dscores, cache_scores) # 2nd layer backprop
    dW2 += self.reg * W2                                   # adding regularization

    #dx, dW1, db1 = affine_relu_backward(dx1, cache_HL) # 1st layer backprop
    #99999999999999999999999999999999999999999999999999999999999999999999999999999999999999
    dx, dW1, db1 = affine_leakyrelu_backward(dx1, cache_HL) # 1st layer backprop
    
    
    dW1 += self.reg * W1                               # adding regularization

    grads.update({'W1': dW1,'b1': db1,'W2': dW2,'b2': db2})
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None,act_func = 'relu'):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.count = 0                     ##########################
    self.act_func = act_func           
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    # print 'self.num_layers =',self.num_layers
    self.dtype = dtype
    self.params = {}
    #ff = range(self.num_layers-1, 0, -1)
    #print ff
    # print ff[i] for i<len(ff)

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
           
    #self.L = len(hidden_dims) + 1            # L=no of layers = no of HL + 1(output layer)
    #self.N = input_dim
    #self.C = num_classes
    
    dimensions = [input_dim] + hidden_dims + [num_classes]     # input_dim + hidden_dim + no_classes
    
    for i in range(1,self.num_layers+1):
        self.params['b%d' % (i)] = np.zeros(dimensions[i])                  # setting biases for the i+1 layer to 0
        self.params['W%d' % (i)] = weight_scale * np.random.randn(dimensions[i-1], dimensions[i]) # setting weights for the i+1 layer
        
        # incorporating batch normalization 
        if self.use_batchnorm and i < len([input_dim] + hidden_dims):
            
            # initializing scale parameter gamma to 1
            #print 'magical quantity gamma %d='%i,(hidden_dims + [num_classes])[i-1]
            self.params['gamma%i'%(i)] = np.ones((hidden_dims + [num_classes])[i-1])
            
            # initializing shift parameter beta to 0
            self.params['beta%i'%(i)] = np.zeros((hidden_dims + [num_classes])[i-1])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
        self.dropout_param['mode'] = mode
    
    if self.use_batchnorm:
        for bn_param in self.bn_params:
            bn_param['mode'] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    
    cache_layer = {}
    inp = X
    for no_layer in xrange(1, self.num_layers):
        if self.use_batchnorm:
   
            # forward pass through affine layer
            inp1, cache1 = affine_forward(inp,self.params['W%i'%no_layer],self.params['b%i'%no_layer])
            
            # batch normalization
            inp2, cache2 = batchnorm_forward(inp1, self.params['gamma%i'%no_layer],
                                               self.params['beta%i'%no_layer],self.bn_params[no_layer-1])
            
            # forward pass through relu
            if self.act_func=='relu':
                out, cache3 = relu_forward(inp2)
            # forward pass through leakyrelu
            if self.act_func=='leakyrelu':
                out, cache3 = leakyrelu_forward(inp2)
            # forward pass through softplus
            if self.act_func=='softplus':
                out, cache3 = softplus_forward(inp2)
            # forward pass through elu
            if self.act_func=='elu':
                out, cache3 = elu_forward(inp2)    
            if self.act_func=='selu':
                out, cache3 = selu_forward(inp2)    
                   
               
            
            cache = (cache1, cache2, cache3)
            intermediate_inp, intermediate_cachei = out,cache
            
        else:
            #9999999999999999999999999999999999999999999999999999999999999999999999999999
            if self.act_func=='relu':
                intermediate_inp, intermediate_cachei = affine_relu_forward(inp, self.params['W%i'%no_layer], self.params['b%i'%no_layer])
            if self.act_func=='leakyrelu':
                intermediate_inp, intermediate_cachei=affine_leakyrelu_forward(inp,self.params['W%i'%no_layer],self.params['b%i'%no_layer])
            if self.act_func=='softplus':
                intermediate_inp, intermediate_cachei= affine_softplus_forward(inp,self.params['W%i'%no_layer],self.params['b%i'%no_layer])
            if self.act_func=='elu':
                intermediate_inp, intermediate_cachei = affine_elu_forward(inp, self.params['W%i'%no_layer],self.params['b%i'%no_layer])
            if self.act_func=='selu':
                intermediate_inp, intermediate_cachei = affine_selu_forward(inp, self.params['W%i'%no_layer],self.params['b%i'%no_layer])

            

        if self.use_dropout:
            inp, d_cache = dropout_forward(intermediate_inp, self.dropout_param)
            cachei = (d_cache, intermediate_cachei)
        else:
            inp = intermediate_inp
            cachei = intermediate_cachei

        cache_layer[no_layer] = cachei
    # The last layer is just an affine layer

    scores, cachei = affine_forward(inp, self.params['W%i'%(self.num_layers)], self.params['b%i'%(self.num_layers)])
    cache_layer[self.num_layers] = cachei
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
        return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dlast = softmax_loss(scores, y)
    regterms = [np.sum(self.params['W%i'%i] * self.params['W%i'%i])
                      for i in xrange(1, self.num_layers+1)]
    loss += 0.5*self.reg * np.sum(regterms)

    ######################################################################

    dl, dw, db = affine_backward(dlast, cache_layer[self.num_layers])
    dw += self.reg*self.params['W%i'%(self.num_layers)]
    db += self.reg*self.params['b%i'%(self.num_layers)]
    grads['W%i'%(self.num_layers)] = dw                    # updation for last layer
    grads ['b%i'%(self.num_layers)] = db

    for layer in xrange(self.num_layers-1, 0, -1):         # updation for all other layers one by one
        if self.use_dropout:
            d_cache, intermediate_cachei = cache_layer[layer]
            dl = dropout_backward(dl, d_cache)
        else:
            intermediate_cachei = cache_layer[layer]

        if self.use_batchnorm:
            if self.act_func=='relu':
                dl, dw, db, dgamma, dbeta = affine_norm_relu_backward(dl, intermediate_cachei)
            if self.act_func=='leakyrelu':
                dl, dw, db, dgamma, dbeta = affine_norm_leakyrelu_backward(dl, intermediate_cachei)
            if self.act_func=='softplus':
                dl, dw, db, dgamma, dbeta = affine_norm_softplus_backward(dl, intermediate_cachei)
            if self.act_func=='elu':
                dl, dw, db, dgamma, dbeta = affine_norm_elu_backward(dl, intermediate_cachei)
            if self.act_func=='selu':
                dl, dw, db, dgamma, dbeta = affine_norm_selu_backward(dl, intermediate_cachei)
            grads['gamma%i'%layer] = dgamma
            grads['beta%i'%layer] = dbeta
        else:
            
            #9999999999999999999999999999999999999999999999999999999999999999999999999999999999
            #dl, dw, db = affine_relu_backward(dl, intermediate_cachei)
            #dl, dw, db = affine_leakyrelu_backward(dl, intermediate_cachei)
            #dl, dw, db = affine_softplus_backward(dl, intermediate_cachei)
            #dl, dw, db = affine_elu_backward(dl, intermediate_cachei)
            dl, dw, db = affine_selu_backward(dl, intermediate_cachei)
        dw += self.reg*self.params['W%i'%layer]
        
        ############################updateeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
        
        grads['W%i'%layer] = dw
        grads['b%i'%layer] = db
        
        '''
        if np.random.uniform()>0.5:     # > x implies 100-x % chance of update
            #print 'updating weights and biases'
            grads['W%i'%layer] = dw
            grads['b%i'%layer] = db
        else:
            #print 'not updating weights and biases'
            grads['W%i'%layer] = np.zeros(dw.shape)            # in case of no update
            grads['b%i'%layer] = np.zeros(db.shape)            # in case of no update             
        '''
        #self.count += 1 ###########################     no of times the loss function is called and parameters are updated
        #print self.count #####################
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

# I havent used this function in the forward pass in the loss function.
def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma, beta, bn_params: params for the batch normalization
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a1, fc_cache = affine_forward(x, w, b)
  a2, norm_cache = batchnorm_forward(a1, gamma, beta, bn_param)
  out, relu_cache = relu_forward(a2)
  cache = (fc_cache, norm_cache, relu_cache)
  return out, cache

# I have used this function for backward pass in the loss function.
def affine_norm_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, norm_cache, relu_cache = cache
  da1 = relu_backward(dout, relu_cache)
  da2, dgamma, dbeta = batchnorm_backward(da1, norm_cache)
  dx, dw, db = affine_backward(da2, fc_cache)
  return dx, dw, db, dgamma, dbeta
            
def affine_norm_leakyrelu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, norm_cache, relu_cache = cache
  da1 = leakyrelu_backward(dout, relu_cache)
  da2, dgamma, dbeta = batchnorm_backward(da1, norm_cache)
  dx, dw, db = affine_backward(da2, fc_cache)
  return dx, dw, db, dgamma, dbeta
                    
def affine_norm_softplus_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, norm_cache, relu_cache = cache
  da1 = softplus_backward(dout, relu_cache)
  da2, dgamma, dbeta = batchnorm_backward(da1, norm_cache)
  dx, dw, db = affine_backward(da2, fc_cache)
  return dx, dw, db, dgamma, dbeta
                    
def affine_norm_elu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, norm_cache, relu_cache = cache
  da1 = elu_backward(dout, relu_cache)
  da2, dgamma, dbeta = batchnorm_backward(da1, norm_cache)
  dx, dw, db = affine_backward(da2, fc_cache)
  return dx, dw, db, dgamma, dbeta

def affine_norm_selu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, norm_cache, relu_cache = cache
  da1 = selu_backward(dout, relu_cache)
  da2, dgamma, dbeta = batchnorm_backward(da1, norm_cache)
  dx, dw, db = affine_backward(da2, fc_cache)
  return dx, dw, db, dgamma, dbeta
    