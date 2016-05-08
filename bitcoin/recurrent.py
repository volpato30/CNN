#!/opt/sharcnet/python/2.7.5/gcc/bin/python
from __future__ import print_function
import numpy as np
import time
import theano
import theano.tensor as T
import lasagne
from label_data import label_data
from iterate_minibatch import iterate_minibatches
from lasagne.regularization import regularize_layer_params, l2, l1


lamda = 0.1

WINDOW = 50

N_HIDDEN = 100
# Number of training sequences in each batch
N_BATCH = 10000
# Optimization learning rate
LEARNING_RATE = .01
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?

NUM_EPOCHS = 500


a = np.load("/scratch/rqiao/okcoin/labeled-02-12:18.npz")
data = a['arr_0']
timestamp = a['arr_1']
label = a['arr_2']
#scale price:
priceIndex = np.linspace(0,18,10,dtype=np.int8)
price = data[:,priceIndex]
meanPrice = price.mean()
stdPrice = price.std()
price = (price-meanPrice)/stdPrice
data[:,priceIndex] = price
volumeIndex = np.linspace(1,19,10,dtype=np.int8)
for index in volumeIndex:
    volume = data[:,index]
    meanVolume = volume.mean()
    stdVolume = volume.std()
    volume = (volume-meanVolume)/stdVolume
    data[:,index] = volume
#data split
train_data, train_label = data[:-20200,:20], label[:-20200]
valid_data, valid_label = data[-20200:-10100,:20], label[-20200:-10100]
test_data, test_label = data[-10100:,:20], label[-10100:]

def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, WINDOW, 20))

    l_forward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True, backwards=True)
    # Now, we'll concatenate the outputs to combine them.
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
    # Our output layer is a simple dense connection, with 1 output unit
    l_out = lasagne.layers.DenseLayer(
        l_concat, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)

    target_values = T.ivector('target_output')

    prediction = lasagne.layers.get_output(l_out)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_values)
    l1_penalty = regularize_layer_params(l_out, l1)
    test_loss = loss.mean()
    loss = test_loss + lamda * l1_penalty
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_values),dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(l_out)
    LEARNING_RATE = .01
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(loss, all_params,LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values],
                            loss, updates=updates)
    valid = theano.function([l_in.input_var, target_values],
                            [test_loss, acc])
    accuracy = theano.function(
        [l_in.input_var, target_values],acc )

    result = theano.function([l_in.input_var],prediction)

    best_acc=0

    print("Training ...")
    try:
        for epoch in range(NUM_EPOCHS):
            if epoch % 40 == 39:
                LEARNING_RATE *= 0.5
                updates = lasagne.updates.adagrad(loss, all_params, LEARNING_RATE)
                train = theano.function([l_in.input_var, target_values],
                                        loss, updates=updates)
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
