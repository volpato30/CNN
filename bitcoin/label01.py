#!/opt/sharcnet/python/2.7.5/gcc/bin/python

from __future__ import print_function
import numpy as np
import time
import theano
import theano.tensor as T
import lasagne
from label_data import label_data
from iterate_minibatch import iterate_minibatches

timestep = 3000
margin = 0.08

a = np.load("/scratch/rqiao/okcoin/2016-01.npz")
data = a['arr_0']
timestamp = a['arr_1']
label = label_data(data, timestamp, timestep, margin)
data = data[:len(label)]
np.savez("/scratch/rqiao/okcoin/labeled2016-01t30.npz",data,timestamp,label)
