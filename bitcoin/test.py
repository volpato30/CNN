#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import numpy as np
a = np.load("/scratch/rqiao/okcoin/2016-01.npz")
data = a['arr_0']
timestamp = a['arr_1']
print data.shape
print timestamp.shape
