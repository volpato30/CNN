#!/opt/sharcnet/python/2.7.5/gcc/bin/python

from __future__ import print_function
import numpy as np
from label_data import label_data
import scipy.io as sio
import numpy as np
import os
import re
timestep = 3000
margin = 0.08

a = np.load("/scratch/rqiao/okcoin/2016-02.npz")
data = a['arr_0']
timestamp = a['arr_1']
print(data.shape)
label = label_data(data, timestamp, timestep, margin)
data = data[:len(label)]
timestamp = timestamp[:len(label)]
np.savez("/scratch/rqiao/okcoin/labeled-02.npz",data,timestamp,label)
print(data.shape)
print(timestamp.shape)
print(label.shape)
