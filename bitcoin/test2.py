#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import numpy as np
a = np.load("/scratch/rqiao/okcoin/labeled2016-01t30.npz")
data = a['arr_0']
timestamp = a['arr_1']
label = a['arr_2']
print data.shape
print timestamp.shape
print label.shape
