#!/opt/sharcnet/python/2.7.5/gcc/bin/python

from __future__ import print_function
import numpy as np
from label_data import label_data
import scipy.io as sio
import numpy as np
import os
import re
import glob

timestep = 3000
margin = 0.08

os.chdir('/scratch/rqiao/okcoin/2016-02')
fname=glob.glob('*.mat')

data=np.zeros((1,80),dtype=np.float32)
timestamp = np.zeros(1,dtype = np.float64)
for file in fname[12:19]:
    temp=sio.loadmat(file)
    data=np.concatenate((data,temp['orderbook']),axis=0)
    timestamp=np.concatenate((timestamp,temp['timestamp'].flatten()),axis=0)
data = data[1:,:]
print(data.shape)
timestamp = timestamp[1:]
label = label_data(data, timestamp, timestep, margin)
data = data[:len(label)]
timestamp = timestamp[:len(label)]
np.savez("/scratch/rqiao/okcoin/labeled-02-12:18.npz",data,timestamp,label)
print(data.shape)
print(timestamp.shape)
print(label.shape)
