
# coding: utf-8

# In[1]:

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

batch_size = 40
Conv2DLayer = lasagne.layers.Conv2DLayer

def load_data():
    
    face=sklearn.datasets.fetch_olivetti_faces(shuffle=True)
    train_set=(face.data[0:200,].reshape((200,1,64,64)),face.target[0:200,].astype(np.int32))
    test_set =(face.data[200:400,].reshape((200,1,64,64)),face.target[200:400,].astype(np.int32))
    rval = [train_set, test_set]
    return rval



def inception_module(l_in,pool_filters, num_1x1, reduce_3x3, num_3x3, reduce_5x5, num_5x5, bias=0):
    """
    inception module (without the 3x3s1 pooling and projection because that's difficult in Theano right now)
    """
    out_layers = []

    if pool_filters > 0:
        l_pool = lasagne.layers.MaxPool2DLayer(l_in, pool_size=3, stride=1, pad=1)
        l_pool_reduced = lasagne.layers.NINLayer(l_pool, num_units=pool_filters,nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(bias))
        out_layers.append(l_pool_reduced)

    # 1x1
    if num_1x1 > 0:
        l_1x1 = lasagne.layers.NINLayer(l_in, num_units=num_1x1,nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(bias))
        out_layers.append(l_1x1)
    
    # 3x3
    if num_3x3 > 0:
        if reduce_3x3 > 0:
            l_reduce_3x3 = lasagne.layers.NINLayer(l_in, num_units=reduce_3x3,nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(bias))
        else:
            l_reduce_3x3 = l_in
        l_3x3 = Conv2DLayer(l_reduce_3x3, num_filters=num_3x3, filter_size=(3, 3), pad="same",nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(bias))
        out_layers.append(l_3x3)
    
    # 5x5
    if num_5x5 > 0:
        if reduce_5x5 > 0:
            l_reduce_5x5 = lasagne.layers.NINLayer(l_in, num_units=reduce_5x5, nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(bias))
        else:
            l_reduce_5x5 = l_in
        l_5x5 = Conv2DLayer(l_reduce_5x5, num_filters=num_5x5, filter_size=(5, 5), pad="same", nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(bias))
        out_layers.append(l_5x5)
    
    # stack
    l_out = lasagne.layers.concat(out_layers)
    return l_out



def build_cnn(input_var=None):
    
    network = lasagne.layers.InputLayer(shape=(batch_size, 1, 64, 64),
                                        input_var=input_var)
   
    network = lasagne.layers.NINLayer(network, num_units=32,nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0))
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)
    
    network = inception_module(
            network,pool_filters=32, num_1x1=64, reduce_3x3=64, num_3x3=64, reduce_5x5=16, num_5x5=32)
    
    network = inception_module(
            network,pool_filters=32, num_1x1=128, reduce_3x3=96, num_3x3=96, reduce_5x5=32, num_5x5=32)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)


    network = inception_module(
            network,pool_filters=32, num_1x1=192, reduce_3x3=96, num_3x3=208, reduce_5x5=16, num_5x5=48)

    network = inception_module(
            network,pool_filters=32, num_1x1=160, reduce_3x3=112, num_3x3=224, reduce_5x5=24, num_5x5=64)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)
    
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    
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
    learnrate=0.02
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    network = build_cnn(input_var)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    l1_penalty = regularize_layer_params(network, l1)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()+0.002*l1_penalty
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.adadelta(loss, params,learning_rate=learnrate)
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
        if epoch % 30 == 29:
            learnrate*= 0.8
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
            np.savez('ORL_inception.npz', *lasagne.layers.get_all_param_values(network))
    return best_acc

    # Optionally, you could now dump the network weights to a file like this:
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


# In[ ]:




# In[5]:

main(500)


# In[ ]:



