#!/opt/sharcnet/python/2.7.5/gcc/bin/python


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
from lasagne.regularization import regularize_layer_params, l2, l1

from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax

batch_size = 40
Conv2DLayer = lasagne.layers.Conv2DLayer
bias = 0

def load_data():

    face=sklearn.datasets.fetch_olivetti_faces(shuffle=True)
    train_set=(face.data[0:200,].reshape((200,1,64,64)),face.target[0:200,].astype(np.int32))
    test_set =(face.data[200:400,].reshape((200,1,64,64)),face.target[200:400,].astype(np.int32))
    rval = [train_set, test_set]
    return rval

def build_cnn(input_var=None):
    net = InputLayer(shape = (None, 1, 64, 64),input_var=input_var)
    net = ConvLayer(
        net, 64, 3, pad=1, flip_filters=False)
    net = ConvLayer(
        net, 64, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = ConvLayer(
        net, 128, 3, pad=1, flip_filters=False)
    net = ConvLayer(
        net, 128, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = ConvLayer(
        net, 256, 3, pad=1, flip_filters=False)
    net = ConvLayer(
        net, 256, 3, pad=1, flip_filters=False)
    net = ConvLayer(
        net, 256, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = DenseLayer(net, num_units=128)
    net = DenseLayer(
        net, num_units=40, nonlinearity=None)
    net = NonlinearityLayer(net, softmax)
    return net

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


# In[4]:

def main(num_epochs=200):
    # Load the dataset
    print("Loading data...")
    datasets = load_data()
    X_train, y_train = datasets[0]
    X_test, y_test = datasets[1]
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    learnrate=0.005
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    network = build_cnn(input_var)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    l2_penalty = regularize_layer_params(network, l2)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + 0.01*l2_penalty
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)

    #updates = lasagne.updates.adadelta(loss, params)
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

    best_acc = 0
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        if epoch % 50 == 49:
            learnrate*=0.8
            #updates = lasagne.updates.adadelta(loss, params,learning_rate=learnrate)
            updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=learnrate, momentum=0.9)
            train_fn = theano.function([input_var, target_var], loss, updates=updates)

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test,batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        test_err = test_err / test_batches
        test_acc = test_acc / test_batches
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  test loss:\t\t{:.6f}".format(test_err))
        print("  validation accuracy:\t\t{:.2f} %".format(
            test_acc * 100))

        if test_acc > best_acc:
            best_acc = test_acc
            np.savez('ORL_Lenet.npz', *lasagne.layers.get_all_param_values(network))
    return best_acc

    # Optionally, you could now dump the network weights to a file like this:
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

final_results = main(500)
print("final accuracy is:\t\t{:.6f}".format(final_results * 100))
