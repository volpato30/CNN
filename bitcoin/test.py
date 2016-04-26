#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import numpy as np
data, timestamp = np.load("/scratch/rqiao/okcoin/2016-01.npz")
print data.shape
print timestamp.shape
