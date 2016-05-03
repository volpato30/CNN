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

os.chdir('/scratch/rqiao/okcoin/2016-02')
file = '2016-02-01.mat'
temp=sio.loadmat(file)
data = temp['orderbook']
print(data.shape)
timestamp = temp['timestamp'].flatten()
label = label_data(data, timestamp, timestep, margin)
data = data[:len(label)]
timestamp = timestamp[:len(label)]
np.savez("/scratch/rqiao/okcoin/labeled0201.npz",data,timestamp,label)
print(data.shape)
print(timestamp.shape)
print(label.shape)
