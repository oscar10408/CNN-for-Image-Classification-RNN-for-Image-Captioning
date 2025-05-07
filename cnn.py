import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn.py!")


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
               hidden_dim=100, num_classes=10, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters_1: Number of filters to use in the first convolutional layer
        - num_filters_2: Number of filters to use in the second convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.dtype = dtype
        (self.C, self.H, self.W) = input_dim
        self.filter_size = filter_size
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Your initializations should work for any valid input dims,      #
        # number of filters, hidden dims, and num_classes. Assume that we use      #
        # max pooling with pool height and width 2 with stride 2.                  #
        #                                                                          #
        # For Linear layers, weights and biases should be initialized from a       #
        # uniform distribution from -sqrt(k) to sqrt(k),                           #
        # where k = 1 / (#input features)                                          #
        # For Conv. layers, weights should be initialized from a uniform           #
        # distribution from -sqrt(k) to sqrt(k),                                   #
        # where k = 1 / ((#input channels) * filter_size^2)                        #
        # Note: we use the same initialization as pytorch.                         #
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html           #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html           #
        #                                                                          #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights for the convolutional layer using the keys 'W1' and 'W2'   #
        # (here we do not consider the bias term in the convolutional layer);      #
        # use keys 'W3' and 'b3' for the weights and biases of the                 #
        # hidden fully-connected layer, and keys 'W4' and 'b4' for the weights     #
        # and biases of the output affine layer.                                   #
        #                                                                          #
        # Make sure you have initialized W1, W2, W3, W4, b3, and b4 in the         #
        # params dicitionary.                                                      #
        #                                                                          #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3. Calculate the size of W3 dynamically           #
        ############################################################################
        
        # 1st Convolution
        self.params['W1'] = np.random.uniform(-np.sqrt(1/(self.C * filter_size**2)),
                                              np.sqrt(1/(self.C * filter_size**2)),
                                              size = (num_filters_1, self.C, filter_size, filter_size))

        output_shape = (num_filters_1, self.H-filter_size+1, self.W-filter_size+1)
        
        # 1st Max Pool
        maxPool_output_shape_H1 = 1 + (output_shape[1] - 2) // 2
        maxPool_output_shape_W1 = 1 + (output_shape[2] - 2) // 2

        output_shape = (num_filters_1, maxPool_output_shape_H1, maxPool_output_shape_W1)

        # 2nd Convolution
        self.params['W2'] = np.random.uniform(-np.sqrt(1/(num_filters_1 * filter_size**2)),
                                              np.sqrt(1/(num_filters_1 * filter_size**2)),
                                              size = (num_filters_2, num_filters_1, filter_size, filter_size))

        output_shape = (num_filters_2, output_shape[1]-filter_size+1, output_shape[2]-filter_size+1)

        # 2nd Max Pool
        maxPool_output_shape_H2 = 1 + (output_shape[1] - 2) // 2
        maxPool_output_shape_W2 = 1 + (output_shape[2] - 2) // 2

        output_shape = (num_filters_2, maxPool_output_shape_H2, maxPool_output_shape_W2)

        # Fully-Connected
        input_size = output_shape[0] * output_shape[1] * output_shape[2] 

        # 1st FC Layer
        self.params["W3"] = np.random.uniform(-np.sqrt(1/ input_size),
                                              np.sqrt(1/ input_size),
                                              size = (input_size, hidden_dim))
        self.params["b3"] = np.zeros(hidden_dim)

        # 2nd FC Layer
        self.params["W4"] = np.random.uniform(-np.sqrt(1/ hidden_dim),
                                              np.sqrt(1/ hidden_dim),
                                              size = (hidden_dim, num_classes))
        self.params["b4"] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3.                                                #
        ############################################################################
        convOut1, convCache1 = conv_forward(X, W1)
        reluOut1, reluCache1 = relu_forward(convOut1)
        maxOut1, maxCache1 = max_pool_forward(reluOut1, pool_param)
        convOut2, convCache2 = conv_forward(maxOut1, W2)
        reluOut2, reluCache2 = relu_forward(convOut2)
        maxOut2, maxCache2 = max_pool_forward(reluOut2, pool_param)
        fcOut1, fcCache1 = fc_forward(maxOut2.reshape(maxOut2.shape[0], -1), W3, b3)
        reluOut3, reluCache3 = relu_forward(fcOut1)
        fcOut2, fcCache2 = fc_forward(reluOut3, W4, b4)
        scores = fcOut2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k].                                                      #
        # Hint: The backwards from W3 needs to be un-flattened before it can be    #
        # passed into the max pool backwards                                       #
        ############################################################################
        loss, dfc2 = softmax_loss(scores, y)
        drelu3, grads['W4'], grads['b4'] = fc_backward(dfc2, fcCache2)
        dfc1 = relu_backward(drelu3, reluCache3)
        dmax2, grads['W3'], grads['b3'] = fc_backward(dfc1, fcCache1)
        dmax2 = dmax2.reshape(maxOut2.shape)
        drelu2 = max_pool_backward(dmax2, maxCache2)
        dconv2 = relu_backward(drelu2, reluCache2)
        dmax1, grads['W2'] = conv_backward(dconv2, convCache2)
        drelu1 = max_pool_backward(dmax1, maxCache1)
        dconv1 = relu_backward(drelu1, reluCache1)
        dx, grads['W1'] = conv_backward(dconv1, convCache1)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
