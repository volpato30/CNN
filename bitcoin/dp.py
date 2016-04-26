#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import glob
import scipy.io as sio
import numpy as np
import os
import re

os.chdir('/scratch/rqiao/okcoin/2016-02')
fname=glob.glob('*.mat')

data=np.zeros((1,80),dtype=np.float32)
time = np.zeros(1,dtype = np.float64)
for file in fname:
    temp=sio.loadmat(file)
    data=np.concatenate((data,temp['orderbook']),axis=0)
    time=np.concatenate((time,temp['timestamp'].flatten()),axis=0)
data = data[1:,:]
time = time[1:]

np.savez("/scratch/rqiao/okcoin/2016-02.npz",data,time)
