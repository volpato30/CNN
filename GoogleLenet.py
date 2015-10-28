#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import sklearn
from sklearn import datasets
import lasagne

def load_data():
    
    face=sklearn.datasets.fetch_olivetti_faces(shuffle=True)
    train_set=(face.data[0:200,].reshape((200,1,64,64)),face.target[0:200,].astype(np.int32))
    valid_set=(face.data[200:300,].reshape((100,1,64,64)),face.target[200:300,].astype(np.int32))
    test_set =(face.data[300:400,].reshape((100,1,64,64)),face.target[300:400,].astype(np.int32))
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an np.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #np.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    rval = [train_set,valid_set, test_set]
    return rval

Conv2DLayer = lasagne.layers.Conv2DLayer

def inception_module(l_in, num_1x1, reduce_3x3, num_3x3, reduce_5x5, num_5x5, gain=1.0, bias=0.1):
    """
    inception module (without the 3x3s1 pooling and projection because that's difficult in Theano right now)
    """
    out_layers = []

    # 1x1
    if num_1x1 > 0:
        l_1x1 = lasagne.layers.NINLayer(l_in, num_units=num_1x1, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_1x1)
    
    # 3x3
    if num_3x3 > 0:
        if reduce_3x3 > 0:
            l_reduce_3x3 = lasagne.layers.NINLayer(l_in, num_units=reduce_3x3, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        else:
            l_reduce_3x3 = l_in
        l_3x3 = Conv2DLayer(l_reduce_3x3, num_filters=num_3x3, filter_size=(3, 3), pad="same", W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_3x3)
    
    # 5x5
    if num_5x5 > 0:
        if reduce_5x5 > 0:
            l_reduce_5x5 = lasagne.layers.NINLayer(l_in, num_units=reduce_5x5, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        else:
            l_reduce_5x5 = l_in
        l_5x5 = Conv2DLayer(l_reduce_5x5, num_filters=num_5x5, filter_size=(5, 5), pad="same", W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_5x5)
    
    # stack
    l_out = lasagne.layers.concat(out_layers)
    return l_out


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(20, 1, 64, 64),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.NINLayer(network, num_units=32, W=lasagne.init.Orthogonal(1.0), b=lasagne.init.Constant(0.1))
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = inception_module(
            network, num_1x1=64, reduce_3x3=96, num_3x3=128, reduce_5x5=16, num_5x5=32,)
    
    network = inception_module(
            network, num_1x1=128, reduce_3x3=128, num_3x3=192, reduce_5x5=32, num_5x5=96,)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = inception_module(
            network, num_1x1=192, reduce_3x3=96, num_3x3=208, reduce_5x5=16, num_5x5=48,)
    
    network = inception_module(
            network, num_1x1=160, reduce_3x3=112, num_3x3=224, reduce_5x5=24, num_5x5=64,)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=40,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network



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


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=100):
    # Load the dataset
    print("Loading data...")
    datasets = load_data()
    X_train, y_train = datasets[0]
    X_val, y_val = datasets[1]
    X_test, y_test = datasets[2]
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    learnrate=0.01
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    network = build_cnn(input_var)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learnrate, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        if epoch % 8 == 7:
            learnrate*=0.96
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learnrate, momentum=0.9)
        for batch in iterate_minibatches(X_train, y_train, 20, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 20, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 20, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    np.savez('model2.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
        main(500)
