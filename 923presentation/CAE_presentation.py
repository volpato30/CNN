import os, sys, urllib, gzip
sys.path.append('/home/rui/pylearn2')
from __future__ import print_function
try:
    import cPickle as pickle
except:
    import pickle
sys.setrecursionlimit(10000)

import matplotlib
matplotlib.use('Agg') # Change matplotlib backend, in case we have no X server running..
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from IPython.display import Image as IPImage
from PIL import Image
from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.nonlinearities import rectify, leaky_rectify, tanh
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy
import pylearn2
from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
from lasagne.regularization import regularize_layer_params, l2, l1
import theano
import theano.tensor as T
import time
import lasagne
import sklearn
from sklearn import datasets
from lasagne.layers import Conv2DLayer as Conv2DLayerSlow
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerSlow
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayerFast
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
    print('Using lasagne.layers (slower)')

face=sklearn.datasets.fetch_olivetti_faces(shuffle=True)
X = face.data
X_out = X
X = np.reshape(X, (-1, 1, 64, 64))
print('X type and shape:', X.dtype, X.shape)
print('X.min():', X.min())
print('X.max():', X.max())

conv_num_filters = 16
filter_size = 3
pool_size = 2
encode_size = 30
dense_mid_size = 256
pad_in = 'valid'
pad_out = 'full'

def build_cnn(input_var=None):

    network = InputLayer(shape=(None,  X.shape[1], X.shape[2], X.shape[3]),input_var=input_var)

    network = Conv2DLayerFast(network, num_filters=conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_in)

    network = Conv2DLayerFast(network, num_filters=conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_in)

    network = MaxPool2DLayerFast(network, pool_size=pool_size)

    network = Conv2DLayerFast(network, num_filters=2*conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_in)

    network = Conv2DLayerFast(network, num_filters=2*conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_in)

    network = Conv2DLayerFast(network, num_filters=2*conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_in)

    network = MaxPool2DLayerFast(network, pool_size=pool_size)

    network = Conv2DLayerFast(network, num_filters=4*conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_in)

    network = MaxPool2DLayerFast(network, pool_size=pool_size)

    network = ReshapeLayer(network, shape =(([0], -1)))

    network = DenseLayer(network, num_units= dense_mid_size, W=lasagne.init.Orthogonal(1.0))

    encode_layer = DenseLayer(network, name= 'encode', num_units= encode_size, W=lasagne.init.Orthogonal(1.0))

    network = DenseLayer(encode_layer, num_units= dense_mid_size, W=lasagne.init.Orthogonal(1.0))

    network = DenseLayer(network, num_units= 1600, W=lasagne.init.Orthogonal(1.0))

    network = ReshapeLayer(network, shape =(([0], 4*conv_num_filters, 5, 5)))

    network = Upscale2DLayer(network, scale_factor = pool_size)

    network = Conv2DLayerFast(network, num_filters=2*conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_out)

    network = Upscale2DLayer(network, scale_factor = pool_size)

    network = Conv2DLayerFast(network, num_filters=2*conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_out)

    network = Conv2DLayerFast(network, num_filters=2*conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_out)

    network = Conv2DLayerFast(network, num_filters=conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_out)

    network = Upscale2DLayer(network, scale_factor = pool_size)

    network = Conv2DLayerSlow(network, num_filters=conv_num_filters, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_out)

    network = Conv2DLayerSlow(network, num_filters=1, W=lasagne.init.Orthogonal(1.0), filter_size=filter_size, pad=pad_out)

    network = ReshapeLayer(network, shape =(([0], -1)))

    return network, encode_layer

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

#Trainning part
num_epochs = 30
input_var = T.tensor4('inputs')
target_var = T.matrix('targets')
learnrate=0.01
# Create neural network model (depending on first command line parameter)
print("Building model and compiling functions...")
network, encode_layer = build_cnn(input_var)
l2_penalty = regularize_layer_params(network, l2)
l1_penalty = regularize_layer_params(network, l1)
reconstructed = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(reconstructed, target_var)
loss = loss.mean()+0.01*l2_penalty+0.01*l1_penalty

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learnrate, momentum=0.9)
train_fn = theano.function([input_var, target_var], loss, updates=updates)
print("Starting training...")


for epoch in range(num_epochs):
    train_err = 0
    train_batches = 0
    start_time = time.time()
    if epoch % 5 == 4:
        learnrate*=0.9
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=learnrate, momentum=0.9)
    for batch in iterate_minibatches(X, X_out, 500, shuffle=False):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

        # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

    # Optionally, you could now dump the network weights to a file like this:
np.savez('CAE_Olivetti.npz', *lasagne.layers.get_all_param_values(network))
