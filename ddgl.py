#!/usr/bin/env python
#This script draws the output of each layer from the trained Googlenet.

from __future__ import print_function

import sys
import os
import time
import pylab
import pickle
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

datasets = load_data()
X_train, y_train = datasets[0]

input_var = T.tensor4('inputs')

inputl = lasagne.layers.InputLayer(shape=(200, 1, 64, 64),input_var=input_var)
# This time we do not apply input dropout, as it tends to work less well
# for convolutional layers.

# Convolutional layer with 32 kernels of size 5x5. Strided and padded
# convolutions are supported as well; see the docstring.
nin1 = lasagne.layers.NINLayer(inputl, num_units=32, W=lasagne.init.Orthogonal(1.0), b=lasagne.init.Constant(0.1))
# Expert note: Lasagne provides alternative convolutional layers that
# override Theano's choice of which implementation to use; for details
# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

# Max-pooling layer of factor 2 in both dimensions:
mp1 = lasagne.layers.MaxPool2DLayer(nin1, pool_size=(2, 2))

# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
incep1 = inception_module(
        mp1, num_1x1=64, reduce_3x3=96, num_3x3=128, reduce_5x5=16, num_5x5=32,)


incep2 = inception_module(
        incep1, num_1x1=128, reduce_3x3=128, num_3x3=192, reduce_5x5=32, num_5x5=96,)

m2 = lasagne.layers.MaxPool2DLayer(incep2, pool_size=(2, 2))

incep3 = inception_module(
            m2, num_1x1=192, reduce_3x3=96, num_3x3=208, reduce_5x5=16, num_5x5=48,)
    
incep4 = inception_module(
            incep3, num_1x1=160, reduce_3x3=112, num_3x3=224, reduce_5x5=24, num_5x5=64,)

m3 = lasagne.layers.MaxPool2DLayer(incep4, pool_size=(2, 2))

# A fully-connected layer of 256 units with 50% dropout on its inputs:
fullconc = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(m3, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

# And, finally, the 10-unit output layer with 50% dropout on its inputs:
network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(fullconc, p=.5),
        num_units=40,
        nonlinearity=lasagne.nonlinearities.softmax)

with np.load('model2.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

f1 = theano.function([input_var], [lasagne.layers.get_output(incep1)])
f2 = theano.function([input_var], [lasagne.layers.get_output(incep2)])
f3 = theano.function([input_var], [lasagne.layers.get_output(incep3)])
f4 = theano.function([input_var], [lasagne.layers.get_output(incep4)])

incep1out=f1(X_train)
incep2out=f2(X_train)
incep3out=f3(X_train)
incep4out=f4(X_train)

i1=np.asarray(incep1out[0])
i2=np.asarray(incep2out[0])
i3=np.asarray(incep3out[0])
i4=np.asarray(incep4out[0])
# print(i1.shape)
#print(nin1out[0][0])
pylab.gray();
pylab.subplot(5, 9, 5) 
pylab.axis('off')
pylab.imshow(X_train[0, 0, :, :])
pylab.subplot(5, 9, 10) 
pylab.axis('off')
pylab.imshow(i1[0,0,:,:])
pylab.subplot(5, 9, 11) 
pylab.axis('off')
pylab.imshow(i1[0,1,:,:])
pylab.subplot(5, 9, 12) 
pylab.axis('off')
pylab.imshow(i1[0,2,:,:])
pylab.subplot(5, 9, 13) 
pylab.axis('off')
pylab.imshow(i1[0,64,:,:])
pylab.subplot(5, 9, 14) 
pylab.axis('off')
pylab.imshow(i1[0,65,:,:])
pylab.subplot(5, 9, 15) 
pylab.axis('off')
pylab.imshow(i1[0,69,:,:])
pylab.subplot(5, 9, 16) 
pylab.axis('off')
pylab.imshow(i1[0,192,:,:])
pylab.subplot(5, 9, 17) 
pylab.axis('off')
pylab.imshow(i1[0,201,:,:])
pylab.subplot(5, 9, 18) 
pylab.axis('off')
pylab.imshow(i1[0,202,:,:])

pylab.subplot(5, 9, 19) 
pylab.axis('off')
pylab.imshow(i2[0,0,:,:])
pylab.subplot(5, 9, 20) 
pylab.axis('off')
pylab.imshow(i2[0,5,:,:])
pylab.subplot(5, 9, 21) 
pylab.axis('off')
pylab.imshow(i2[0,2,:,:])
pylab.subplot(5, 9, 22) 
pylab.axis('off')
pylab.imshow(i2[0,139,:,:])
pylab.subplot(5, 9, 23) 
pylab.axis('off')
pylab.imshow(i2[0,131,:,:])
pylab.subplot(5, 9, 24) 
pylab.axis('off')
pylab.imshow(i2[0,180,:,:])
pylab.subplot(5, 9, 25) 
pylab.axis('off')
pylab.imshow(i2[0,350,:,:])
pylab.subplot(5, 9, 26) 
pylab.axis('off')
pylab.imshow(i2[0,351,:,:])
pylab.subplot(5, 9, 27) 
pylab.axis('off')
pylab.imshow(i2[0,352,:,:])

pylab.subplot(5, 9, 28) 
pylab.axis('off')
pylab.imshow(i3[0,0,:,:])
pylab.subplot(5, 9, 29) 
pylab.axis('off')
pylab.imshow(i3[0,9,:,:])
pylab.subplot(5, 9, 30) 
pylab.axis('off')
pylab.imshow(i3[0,2,:,:])
pylab.subplot(5, 9, 31) 
pylab.axis('off')
pylab.imshow(i3[0,279,:,:])
pylab.subplot(5, 9, 32) 
pylab.axis('off')
pylab.imshow(i3[0,201,:,:])
pylab.subplot(5, 9, 33) 
pylab.axis('off')
pylab.imshow(i3[0,202,:,:])
pylab.subplot(5, 9, 34) 
pylab.axis('off')
pylab.imshow(i3[0,410,:,:])
pylab.subplot(5, 9, 35) 
pylab.axis('off')
pylab.imshow(i3[0,420,:,:])
pylab.subplot(5, 9, 36) 
pylab.axis('off')
pylab.imshow(i3[0,421,:,:])

pylab.subplot(5, 9, 37) 
pylab.axis('off')
pylab.imshow(i4[0,9,:,:])
pylab.subplot(5, 9, 38) 
pylab.axis('off')
pylab.imshow(i4[0,1,:,:])
pylab.subplot(5, 9, 39) 
pylab.axis('off')
pylab.imshow(i4[0,69,:,:])
pylab.subplot(5, 9, 40) 
pylab.axis('off')
pylab.imshow(i4[0,259,:,:])
pylab.subplot(5, 9, 41) 
pylab.axis('off')
pylab.imshow(i4[0,234,:,:])
pylab.subplot(5, 9, 42) 
pylab.axis('off')
pylab.imshow(i4[0,333,:,:])
pylab.subplot(5, 9, 43) 
pylab.axis('off')
pylab.imshow(i4[0,410,:,:])
pylab.subplot(5, 9, 44) 
pylab.axis('off')
pylab.imshow(i4[0,426,:,:])
pylab.subplot(5, 9, 45) 
pylab.axis('off')
pylab.imshow(i4[0,417,:,:])

pylab.show()
