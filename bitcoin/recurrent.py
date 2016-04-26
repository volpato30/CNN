#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import time
import theano
import theano.tensor as T
import lasagne
from label_data import label_data
from iterate_minibatches import iterate_minibatches


WINDOW = 100

N_HIDDEN = 200
# Number of training sequences in each batch
N_BATCH = 5000
# Optimization learning rate
LEARNING_RATE = .01
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?

NUM_EPOCHS = 10

timestep = 3000
margin = 0.08

a = np.load("/scratch/rqiao/okcoin/2016-01.npz")
data = a['arr_0']
timestamp = a['arr_1']
label = label_data(data, timestamp, timestep, margin)
data = data[:len(label)]

#scale price:
priceIndex = np.linspace(0,78,40,dtype=np.int8)
price = data[:,priceIndex]
meanPrice = price.mean()
stdPrice = price.std()
price = (price-meanPrice)/stdPrice
data[:,priceIndex] = price

#data split
train_data, train_label = data[:-40200,:], label[:-40200]
valid_data, valid_label = data[-40200:-20100,:], label[-40200:-20100]
test_data, test_label = data[-20100:,:], label[-20100:]

def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 80))

    l_forward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True, backwards=True)
    # Now, we'll concatenate the outputs to combine them.
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
    # Our output layer is a simple dense connection, with 1 output unit
    l_out = lasagne.layers.DenseLayer(
        l_concat, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)

    target_values = T.vector('target_output')

    prediction = lasagne.layers.get_output(l_out)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_values)
    loss = loss.mean()
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_values),dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(l_out)

    print("Computing updates ...")
    updates = lasagne.updates.adagrad(loss, all_params,learn_rate)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values],
                            loss, updates=updates)
    valid = theano.function([l_in.input_var, target_values],
                            [loss, acc])
    accuracy = theano.function(
        [l_in.input_var, target_values],acc )

    result = theano.function([l_in.input_var],prediction)

    best_acc=0

        print("Training ...")
        try:
            for epoch in range(NUM_EPOCHS):

                train_err = 0
                train_batches = 0
                start_time = time.time()
                for batch in iterate_minibatches(train_data, train_label, N_BATCH, WINDOW):
                    inputs, targets = batch
                    train_err += train(inputs, targets)
                    train_batches += 1

                val_err = 0
                val_acc = 0
                val_batches = 0
                for batch in iterate_minibatches(valid_data, valid_label, N_BATCH, WINDOW):
                    inputs, targets = batch
                    err, acc = valid(inputs, targets)
                    val_err += err
                    val_acc += acc
                    val_batches += 1

                val_acc = val_acc / val_batches
                if val_acc > best_acc:
                    best_acc = val_acc

                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, NUM_EPOCHS, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  validation accuracy:\t\t{:.2f} %".format(
                        val_acc * 100))
        except KeyboardInterrupt:
            pass
if __name__ == '__main__':
    main()
